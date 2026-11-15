"""Recommender for B05: candidates, linear scoring, reranking, route, history, cold start.

Sections 6.3-6.6 of the spec. Until B04 ships ML labels and embeddings, item tasks, stages
and levels come from rules over the source labels, and semantic closeness comes from the
TF-IDF fallback index. Every score is a weighted sum of features in [0, 1]; the contribution
of a feature is returned as a reason code, so the explanation is the model itself.
"""

import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field

from alfarec.config import load_configuration
from alfarec.recsys.baseline import TopicBaseline
from alfarec.recsys.contracts import (
    Clarification,
    Contract,
    Event,
    Item,
    Level,
    Mode,
    NextStep,
    Profile,
    Reason,
    ReasonCode,
    Recommendation,
    RecommendationResponse,
    Stage,
)
from alfarec.recsys.explain import next_step_for
from alfarec.recsys.retrieval import DenseIndex, TfidfIndex
from alfarec.recsys.sequence import DEFAULT_TRANSITIONS_PATH, CourseTransitions
from alfarec.recsys.taxonomy import Taxonomy, match_key
from alfarec.recsys.weights import DEFAULT_WEIGHTS_PATH, load_weights, validate_weights

# Initial hand-set weights (spec 6.3); B06 tunes them on the tune split of the profiles.
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 1.2,
    "task": 2.0,
    "stage": 0.6,
    "level": 0.6,
    "industry": 0.8,
    "format": 0.5,
    "history": 1.0,
    "quality": 0.4,
    "popularity": 0.3,
    "fresh": 0.2,
    "offer": 0.3,
    # Enrolment transitions between courses; tuned on the tune split of next-course cases.
    "sequence": 16.0,
}
CANDIDATES = 300
ROUTE_MAX = 5
QUICK_LIMIT = 4
QUICK_MAX_MIN = 5
WEEK_PLAN_DAYS = 5
WEEK_PLAN_MAX_MIN = 10
STARTER_SIZE = 8
MMR_LAMBDA = 0.75
TOP_K = 5
MAX_PAID_IN_TOP = 1
MAX_SAME_TOPIC_IN_TOP = 2
MIN_FORMATS_IN_TOP = 2
CONFIDENT_SEMANTIC = 0.18
FRESH_DAYS = 365
# A reading time is stated as a fact reason up to this length.
FACT_MAX_MIN = 20

EVENT_WEIGHTS = {"complete": 1.0, "enroll": 0.7, "save": 0.6, "view": 0.4, "dislike": -0.7}
EXCLUDED_EVENTS = {"complete", "known", "dislike"}
HALF_LIFE_DAYS = 30.0

# Spec 6.3: "idea" fits launch materials fully and growth ones at 0.2.
STAGE_FIT: dict[str, dict[str, float]] = {
    "idea": {"idea": 1.0, "launch": 1.0, "operating": 0.5, "growth": 0.2},
    "launch": {"idea": 0.7, "launch": 1.0, "operating": 0.8, "growth": 0.3},
    "operating": {"idea": 0.3, "launch": 0.6, "operating": 1.0, "growth": 0.8},
    "growth": {"idea": 0.1, "launch": 0.3, "operating": 0.8, "growth": 1.0},
}
LEVEL_FIT: dict[str, dict[str, float]] = {
    "beginner": {"beginner": 1.0, "intermediate": 0.6, "advanced": 0.1},
    "intermediate": {"beginner": 0.6, "intermediate": 1.0, "advanced": 0.7},
    "advanced": {"beginner": 0.3, "intermediate": 0.8, "advanced": 1.0},
}
LEVELS: tuple[Level, ...] = ("beginner", "intermediate", "advanced")
FORMAT_FIT: dict[str, dict[str, float]] = {
    "quick": {"quick_answer": 1.0, "tool": 1.0, "template": 0.7, "guide": 0.5},
    "hour": {"guide": 1.0, "template": 1.0, "tool": 0.9, "recording": 0.8, "podcast": 0.8,
             "quick_answer": 0.7, "course": 0.4},
    "course": {"course": 1.0, "recording": 0.8, "podcast": 0.6, "guide": 0.5},
}  # fmt: skip
ROUTE_ROLES = {
    "quick_answer": "understand",
    "guide": "understand",
    "template": "do",
    "tool": "do",
    "course": "deepen",
    "recording": "deepen",
    "podcast": "deepen",
}
START_LABELS = {match_key(v) for v in ("Старт бизнеса", "Первые шаги", "Регистрация бизнеса")}
GROWTH_LABELS = {match_key(v) for v in ("Развитие бизнеса",)}
IDEA_LABELS = {match_key(v) for v in ("Идеи для бизнеса", "Выбор ниши", "Генерация идей")}
LAW_CHANGE_LABELS = {match_key(v) for v in ("Изменения в законах", "Законы", "Налоги и законы")}
BEGINNER_TITLE = re.compile(
    r"с нуля|для начинающ|основы|азбука|азы|просто о|что такое|с чего начать|первые шаги|"
    r"для новичк|как открыть|как начать",
    re.IGNORECASE,
)
ADVANCED_TITLE = re.compile(r"\bLBO\b|масштаб|стратеги|финансовая модель|продвинут", re.IGNORECASE)
_WORD_START = r"(?<![0-9a-zа-я])"


class Boundary(Contract):
    reason: str
    cues: list[str] = Field(min_length=1)
    question: str
    # The boundary fires only together with one of these tasks and none of the exceptions.
    requires_tasks: list[str] = Field(default_factory=list)
    unless_tasks: list[str] = Field(default_factory=list)


class IntentLexicon(Contract):
    version: Literal[1]
    tasks: dict[str, list[str]]
    stages: dict[Stage, list[str]]
    boundaries: list[Boundary]


def load_intents(path: Path | None = None) -> IntentLexicon:
    return load_configuration(path or Path(__file__).with_name("intent.yaml"), IntentLexicon)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def _hits(text: str, cues: Iterable[str]) -> int:
    """Number of cues found; a cue is a regular expression anchored at a word start."""
    return sum(1 for cue in cues if re.search(_WORD_START + cue, text) is not None)


@dataclass
class Understanding:
    tasks: list[str]
    stage: Stage | None
    boundary: Boundary | None
    has_signal: bool


def understand(text: str | None, lexicon: IntentLexicon) -> Understanding:
    """Read tasks, a stage and scope boundaries from free text with explicit cues."""
    if not text or not text.strip():
        return Understanding([], None, None, False)
    query = _normalize(text)
    scores = {task: _hits(query, cues) for task, cues in lexicon.tasks.items()}
    found = {task for task, score in scores.items() if score}
    best = max(scores.values(), default=0)
    tasks = sorted((t for t, s in scores.items() if s and s == best), key=lambda t: t)
    boundary = next(
        (
            b
            for b in lexicon.boundaries
            if _hits(query, b.cues)
            and (not b.requires_tasks or found & set(b.requires_tasks))
            and not found & set(b.unless_tasks)
        ),
        None,
    )
    stage_hits = {stage: _hits(query, cues) for stage, cues in lexicon.stages.items()}
    top = max(stage_hits.values(), default=0)
    leaders = [stage for stage, hits in stage_hits.items() if hits == top]
    # A tie between stages says nothing reliable about the stage.
    stage = leaders[0] if top and len(leaders) == 1 else None
    return Understanding(tasks[:2], stage, boundary, best > 0)


@dataclass(frozen=True)
class ItemFacts:
    """Labels restored from source fields until B04 provides ML labels."""

    tasks: Mapping[str, float]
    stages: frozenset[str]
    level: Level
    industries: frozenset[str]
    topic: str


def _item_facts(item: Item, taxonomy: Taxonomy) -> ItemFacts:
    topics = {match_key(v) for v in item.topics}
    tags = {match_key(v) for v in item.raw_tags}
    subject = match_key(item.subject) if item.subject and item.item_type == "article" else None
    tasks: dict[str, float] = {}
    for task_id, task in taxonomy.tasks.items():
        links = task.source_links
        strength = 0.0
        if tags & {match_key(v) for v in links.tags}:
            strength = 1.0
        if subject and subject in {match_key(v) for v in links.article_subjects}:
            strength = max(strength, 0.8)
        if topics & {match_key(v) for v in links.topics}:
            # Broad topics are shared by several tasks: a weak signal (labeling.md).
            strength = max(strength, 0.5)
        if strength:
            tasks[task_id] = strength
    labels = topics | tags | ({subject} if subject else set())
    raw_industries = {match_key(v) for v in item.raw_industries}
    stages: set[str] = set()
    if labels & START_LABELS or match_key("Без опыта") in raw_industries:
        stages |= {"idea", "launch"}
    if labels & IDEA_LABELS:
        stages.add("idea")
    if labels & GROWTH_LABELS:
        stages |= {"operating", "growth"}
    # Terms and calculators need no preparation: they answer one question at once.
    if item.item_type in {"glossary", "calculator"} or BEGINNER_TITLE.search(item.title):
        level: Level = "beginner"
    elif match_key("С опытом") in raw_industries or ADVANCED_TITLE.search(item.title):
        level = "advanced"
    elif labels & START_LABELS or match_key("Без опыта") in raw_industries:
        level = "beginner"
    else:
        level = "intermediate"
    industries = frozenset(
        industry_id
        for industry_id, industry in taxonomy.industries.items()
        if raw_industries & {match_key(v) for v in industry.raw_values}
    )
    topic = (item.topics[0] if item.topics else item.subject or item.item_type).casefold()
    return ItemFacts(tasks, frozenset(stages), level, industries, topic)


def _item_text(item: Item) -> str:
    return f"{item.title}. {item.summary}"


@dataclass
class Scored:
    item: Item
    score: float
    contributions: dict[str, float] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)
    continuing: bool = False


@dataclass
class _Context:
    profile: Profile
    tasks: list[str]
    stage: Stage | None
    level: Level | None
    level_raised: bool
    history_weights: dict[str, float]
    history_titles: dict[str, str]
    seen_topics: set[str]
    completed_free_tasks: set[str]
    excluded: set[str]
    semantic: dict[str, float]
    semantic_top: float
    neighbours: dict[str, tuple[float, str]]
    empty: bool
    sequence: dict[str, tuple[float, str]] = field(default_factory=dict)
    stated_tasks: bool = False


class Ranker:
    """Implements the Recommender protocol over the catalog and the retrieval index."""

    def __init__(
        self,
        catalog: Sequence[Item],
        taxonomy: Taxonomy,
        index: TfidfIndex | DenseIndex,
        weights: Mapping[str, float] | None = None,
        lexicon: IntentLexicon | None = None,
        now: datetime | None = None,
        transitions: CourseTransitions | None = None,
        rerank: bool = True,
    ) -> None:
        if len({item.item_id for item in catalog}) != len(catalog):
            raise ValueError("Catalog item IDs must be unique")
        missing = {item.item_id for item in catalog} - set(index.item_ids)
        if missing:
            raise ValueError(f"Index does not cover {len(missing)} catalog items")
        self._items = {item.item_id: item for item in catalog}
        self._order = sorted(self._items)
        self._taxonomy = taxonomy
        self._index = index
        self._weights = {
            **DEFAULT_WEIGHTS,
            **validate_weights(weights if weights is not None else {}),
        }
        self._lexicon = lexicon or load_intents()
        self._transitions = transitions
        # Off only for the B06 ablation: the list then keeps the plain score order.
        self._rerank_enabled = rerank
        self._baseline = TopicBaseline(catalog, taxonomy)
        self._facts = {
            item_id: _item_facts(item, taxonomy) for item_id, item in self._items.items()
        }
        self._graph = None
        dates = [item.published_at for item in catalog if item.published_at]
        # A fixed clock keeps answers deterministic; the newest material is "now".
        self._now = now or (max(dates) if dates else datetime(2026, 11, 1, tzinfo=UTC))
        popular = [item.popularity or 0.0 for item in catalog]
        self._max_log_pop = math.log1p(max(popular, default=0.0)) or 1.0

    # Public API -------------------------------------------------------------------------

    def recommend(
        self,
        profile: Profile,
        history: Sequence[Event],
        mode: Mode = "ours",
        limit: int = 10,
    ) -> RecommendationResponse:
        if mode not in ("ours", "baseline"):
            raise ValueError("Unsupported recommendation mode")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Limit must be an integer from 1 to 50")
        if mode == "baseline":
            return self._baseline.recommend(profile, history, mode, limit)
        understanding = understand(profile.free_text, self._lexicon)
        context = self._context(profile, history, understanding)
        if understanding.boundary is not None and not profile.tasks:
            return self._clarify(profile, context, understanding.boundary.question, limit)
        if context.empty and not any(w > 0 for w in context.history_weights.values()):
            return self._cold_start(profile, context, limit)
        scored = self._score(self._candidates(context), context)
        if self._low_confidence(context, understanding):
            return self._clarify(profile, context, "Уточните, что ближе", limit, scored)
        continuing = self._continue_course(history, context)
        prefix = [continuing] if continuing else []
        used = {entry.item.item_id for entry in prefix}
        route = self._route([entry for entry in scored if entry.item.item_id not in used], context)
        used |= {entry.item.item_id for entry in route}
        quick = [
            entry
            for entry in scored
            if entry.item.item_id not in used and self._is_quick(entry.item)
        ][:QUICK_LIMIT]
        used |= {entry.item.item_id for entry in quick}
        # Quotas apply to the list the user actually sees first.
        rest = [entry for entry in scored if entry.item.item_id not in used]
        size = limit - (continuing is not None)
        ranked = self._rerank(rest, size, prefix) if self._rerank_enabled else rest[:size]
        more = prefix + ranked
        return RecommendationResponse(
            route=[self._to_recommendation(entry, context) for entry in route],
            more=[self._to_recommendation(entry, context) for entry in more],
            quick=[self._to_recommendation(entry, context) for entry in quick],
            mode=mode,
            profile_used=self._profile_used(profile, context),
        )

    # Context ----------------------------------------------------------------------------

    def _context(
        self, profile: Profile, history: Sequence[Event], understanding: Understanding
    ) -> _Context:
        tasks = list(dict.fromkeys([*profile.tasks, *understanding.tasks]))
        stated_tasks = bool(tasks)
        weights: dict[str, float] = {}
        titles: dict[str, str] = {}
        excluded: set[str] = set()
        known = 0
        completed_free_tasks: set[str] = set()
        reference = max([event.ts for event in history], default=self._now)
        for event in history:
            if event.item_id not in self._items:
                continue
            if event.type in EXCLUDED_EVENTS:
                excluded.add(event.item_id)
            if event.type == "known":
                known += 1
                continue
            weight = EVENT_WEIGHTS.get(event.type, 0.0)
            if event.type == "rate" and event.value is not None:
                weight = 0.3 if event.value >= 4 else -0.7 if event.value <= 2 else 0.0
            age = max((reference - event.ts).total_seconds() / 86400, 0.0)
            weight *= 0.5 ** (age / HALF_LIFE_DAYS)
            weights[event.item_id] = weights.get(event.item_id, 0.0) + weight
            item = self._items[event.item_id]
            if weight > 0:
                titles[event.item_id] = item.title
            if event.type == "complete" and item.access == "free":
                completed_free_tasks |= set(self._facts[event.item_id].tasks)
        if not tasks and weights:
            # An empty profile is completed from the history (spec 6.3, step 1).
            counter = Counter(
                task
                for item_id, weight in weights.items()
                if weight > 0
                for task, strength in self._facts[item_id].tasks.items()
                if strength >= 0.8
            )
            tasks = [task for task, _ in counter.most_common(2)]
        level = profile.experience
        raised = False
        if known and level != "advanced":
            # "I already know this" moves the user one step up; unknown experience counts as
            # beginner.
            level = LEVELS[LEVELS.index(level or "beginner") + 1]
            raised = True
        query = self._query_text(profile, tasks)
        semantic, semantic_top = self._semantic(query)
        neighbours = self._neighbours(weights)
        sequence = self._sequence(history, excluded | set(weights))
        empty = not (
            tasks or profile.stage or profile.industries or profile.experience
            or profile.time_budget or (profile.free_text and profile.free_text.strip())
        )  # fmt: skip
        return _Context(
            profile=profile,
            tasks=tasks,
            stage=profile.stage or understanding.stage,
            level=level,
            level_raised=raised,
            history_weights=weights,
            history_titles=titles,
            seen_topics={self._facts[i].topic for i in weights},
            completed_free_tasks=completed_free_tasks,
            excluded=excluded,
            semantic=semantic,
            semantic_top=semantic_top,
            neighbours=neighbours,
            empty=empty,
            sequence=sequence,
            stated_tasks=stated_tasks,
        )

    def _query_text(self, profile: Profile, tasks: list[str]) -> str:
        parts = [profile.free_text or ""]
        for task_id in tasks:
            task = self._taxonomy.tasks.get(task_id)
            if task:
                parts += [task.label, task.description]
        for industry_id in profile.industries:
            industry = self._taxonomy.industries.get(industry_id)
            if industry:
                parts.append(industry.label)
        return " ".join(part for part in parts if part).strip()

    def _semantic(self, query: str) -> tuple[dict[str, float], float]:
        """Closeness to the query, scaled so the best candidate scores 1, and the raw best."""
        if not query:
            return {}, 0.0
        hits = self._index.search(query, limit=CANDIDATES)
        top = hits[0].score if hits else 0.0
        return ({hit.item_id: hit.score / top for hit in hits} if top > 0 else {}), top

    def _neighbours(self, weights: Mapping[str, float]) -> dict[str, tuple[float, str]]:
        result: dict[str, tuple[float, str]] = {}
        for item_id, weight in sorted(weights.items()):
            if weight <= 0:
                continue
            source = self._items[item_id]
            if self._graph is not None:
                row = self._graph.getrow(self._graph_positions[item_id])
                for position, similarity in zip(row.indices, row.data, strict=True):
                    target = self._graph_ids[position]
                    value = min(float(similarity) * weight, 1.0)
                    if value > result.get(target, (0.0, ""))[0]:
                        result[target] = (value, source.title)
            for hit in self._item_neighbours(source):
                value = min(hit.score * weight, 1.0)
                if value > result.get(hit.item_id, (0.0, ""))[0]:
                    result[hit.item_id] = (value, source.title)
        for item_id, weight in weights.items():
            if weight < 0:
                source = self._items[item_id]
                for hit in self._item_neighbours(source):
                    value, title = result.get(hit.item_id, (0.0, ""))
                    result[hit.item_id] = (value + hit.score * weight, title)
        return result

    def _item_neighbours(self, source: Item):
        if isinstance(self._index, DenseIndex):
            return self._index.neighbours(source.item_id, limit=30)
        return self._index.search(_item_text(source), limit=30, excluded={source.item_id})

    def _sequence(self, history: Sequence[Event], seen: set[str]) -> dict[str, tuple[float, str]]:
        """Courses people enrol in after the user's courses (enrolment transitions)."""
        if self._transitions is None:
            return {}
        taken = [
            e.item_id
            for e in sorted(history, key=lambda e: (e.ts, e.item_id))
            if e.type in {"enroll", "complete"}
            and e.item_id in self._items
            and self._items[e.item_id].format == "course"
        ]
        candidates = {
            item_id: float(item.popularity or 0)
            for item_id, item in self._items.items()
            if item.format == "course" and item_id not in seen and (item.popularity or 0) > 0
        }
        return self._transitions.scores(taken, candidates)

    # Candidates and scoring ---------------------------------------------------------------

    def _candidates(self, context: _Context) -> list[str]:
        ids = set(context.semantic) & set(self._items)
        ids |= {
            item_id
            for item_id, facts in self._facts.items()
            if any(facts.tasks.get(task, 0) >= 0.5 for task in context.tasks)
        }
        ids |= set(context.neighbours)
        ids |= {
            item_id
            for item_id, item in self._items.items()
            if item.format == "course" and (item.popularity or 0) > 0
        }
        return sorted(ids - context.excluded)

    def _features(self, item: Item, context: _Context) -> tuple[dict[str, float], dict]:
        facts = self._facts[item.item_id]
        features: dict[str, float] = {}
        params: dict[str, dict] = {}
        features["semantic"] = context.semantic.get(item.item_id, 0.0)
        task_strengths = [(facts.tasks.get(t, 0.0), t) for t in context.tasks]
        strength, task = max(task_strengths, default=(0.0, ""))
        features["task"] = strength
        if strength:
            params["task"] = {"task": task}
        if context.stage:
            if facts.stages:
                features["stage"] = max(STAGE_FIT[context.stage][s] for s in facts.stages)
            else:
                features["stage"] = 0.5
            params["stage"] = {"stage": context.stage}
        if context.level:
            features["level"] = LEVEL_FIT[context.level][facts.level]
        if context.profile.industries:
            matched = sorted(facts.industries & set(context.profile.industries))
            features["industry"] = 1.0 if matched else 0.6 if not facts.industries else 0.0
            if matched:
                params["industry"] = {"industry": matched[0]}
        if context.profile.time_budget:
            features["format"] = FORMAT_FIT[context.profile.time_budget].get(item.format, 0.2)
        neighbour, title = context.neighbours.get(item.item_id, (0.0, ""))
        features["history"] = max(min(neighbour, 1.0), -1.0)
        if title:
            params["history"] = {"title": title}
        if item.item_id in context.sequence:
            # With tasks stated by the user, behaviour only ranks courses within those tasks;
            # otherwise a popular follow-up course would push the user's own task aside.
            value = context.sequence[item.item_id][0]
            features["sequence"] = value * strength if context.stated_tasks else value
        quality = []
        if item.completion_rate is not None:
            # Smoothed toward the catalog-wide mean of about 0.12 for courses.
            count = item.popularity or 0.0
            quality.append((item.completion_rate * count + 0.12 * 50) / (count + 50) / 0.3)
        if item.rating is not None:
            quality.append((item.rating - 1) / 4)
        features["quality"] = min(max(quality, default=0.0), 1.0)
        features["popularity"] = math.log1p(item.popularity or 0.0) / self._max_log_pop
        if item.published_at and (self._now - item.published_at) <= timedelta(days=FRESH_DAYS):
            labels = {match_key(v) for v in [*item.topics, *(filter(None, [item.subject]))]}
            features["fresh"] = 1.0 if labels & LAW_CHANGE_LABELS else 0.0
        if item.access != "free":
            wants_depth = context.profile.time_budget == "course"
            did_free = bool(set(facts.tasks) & context.completed_free_tasks)
            features["offer"] = 1.0 if strength and (wants_depth or did_free) else -1.0
        return features, params

    def _score(self, candidate_ids: Iterable[str], context: _Context) -> list[Scored]:
        weights = dict(self._weights)
        if context.empty:
            # With nothing known about the user, proven materials carry the answer.
            weights["popularity"] = weights["popularity"] * 4
            weights["quality"] = weights["quality"] * 2
        scored = []
        for item_id in candidate_ids:
            item = self._items[item_id]
            features, params = self._features(item, context)
            contributions = {name: weights[name] * value for name, value in features.items()}
            scored.append(Scored(item, sum(contributions.values()), contributions, params))
        return sorted(scored, key=lambda entry: (-entry.score, entry.item.item_id))

    # Reranking, route and blocks -------------------------------------------------------

    def _similarity(self, left: Item, right: Item) -> float:
        if isinstance(self._index, DenseIndex):
            return self._index.similarity(left.item_id, right.item_id)
        a, b = self._facts[left.item_id], self._facts[right.item_id]
        shared = set(a.tasks) & set(b.tasks)
        union = set(a.tasks) | set(b.tasks)
        overlap = len(shared) / len(union) if union else 0.0
        return 0.5 * overlap + 0.3 * (a.topic == b.topic) + 0.2 * (left.format == right.format)

    def _rerank(
        self, scored: list[Scored], size: int, prefix: Sequence[Scored] = ()
    ) -> list[Scored]:
        """MMR with the paid, topic and format quotas of the top five; nothing is dropped."""
        pool = list(scored)
        if not pool or size < 1:
            return []
        top_score = max(abs(entry.score) for entry in pool) or 1.0
        selected = list(prefix)
        similarity_state: dict[str, tuple[int, float]] = {}

        def strongest_overlap(entry: Scored) -> float:
            count, value = similarity_state.get(entry.item.item_id, (0, 0.0))
            for previous in selected[count:]:
                value = max(value, self._similarity(entry.item, previous.item))
            similarity_state[entry.item.item_id] = (len(selected), value)
            return value

        while pool and len(selected) < size + len(prefix):
            in_top = len(selected) < TOP_K
            allowed = [e for e in pool if not in_top or self._fits_top(e, selected)]
            if not allowed:
                # Quotas cannot all hold: keep the paid limit, relax topic and format.
                paid_limit_reached = (
                    sum(e.item.access != "free" for e in selected) >= MAX_PAID_IN_TOP
                )
                allowed = [
                    e
                    for e in pool
                    if not in_top or not paid_limit_reached or e.item.access == "free"
                ]
                if not allowed:
                    break
            best = max(
                allowed[: size * 3],
                key=lambda e: (
                    MMR_LAMBDA * e.score / top_score - (1 - MMR_LAMBDA) * strongest_overlap(e),
                    e.item.item_id,
                ),
            )
            selected.append(best)
            pool.remove(best)
        return selected[len(prefix) :]

    def _fits_top(self, entry: Scored, selected: list[Scored]) -> bool:
        paid = sum(s.item.access != "free" for s in selected)
        if entry.item.access != "free" and paid >= MAX_PAID_IN_TOP:
            return False
        topic = self._facts[entry.item.item_id].topic
        if (
            sum(self._facts[s.item.item_id].topic == topic for s in selected)
            >= MAX_SAME_TOPIC_IN_TOP
        ):
            return False
        formats = {s.item.format for s in selected}
        remaining = TOP_K - len(selected)
        if remaining == 1 and len(formats | {entry.item.format}) < MIN_FORMATS_IN_TOP:
            return False
        return True

    def _route(self, ranked: list[Scored], context: _Context) -> list[Scored]:
        """3-5 steps by role: understand, do, deepen; a missing role is skipped, not padded."""
        if not context.tasks and not (context.profile.free_text or "").strip():
            return []
        relevant = [e for e in ranked[:30] if e.contributions.get("task", 0) > 0]
        roles: dict[str, list[Scored]] = {"understand": [], "do": [], "deepen": []}
        for entry in relevant:
            roles[ROUTE_ROLES[entry.item.format]].append(entry)
        route = roles["understand"][:2] + roles["do"][:1] + roles["deepen"][:1]
        if len(route) < 3:
            route += [e for e in roles["understand"][2:3] + roles["do"][1:2] if e not in route]
        if len(route) < 3:
            return []
        order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        role_order = {"understand": 0, "do": 1, "deepen": 2}
        route.sort(
            key=lambda e: (
                role_order[ROUTE_ROLES[e.item.format]],
                order[self._facts[e.item.item_id].level],
                -e.score,
            )
        )
        return route[:ROUTE_MAX]

    def _is_quick(self, item: Item) -> bool:
        if item.format == "quick_answer":
            return item.duration_min is None or item.duration_min <= QUICK_MAX_MIN
        return item.format == "tool"

    def _continue_course(self, history: Sequence[Event], context: _Context) -> Scored | None:
        """The course in progress comes back first: the main reason to return (spec 6.4)."""
        finished = {e.item_id for e in history if e.type in {"complete", "known", "dislike"}}
        started = [
            e
            for e in sorted(history, key=lambda e: (e.ts, e.item_id), reverse=True)
            if e.type == "enroll"
            and e.item_id in self._items
            and e.item_id not in finished
            and self._items[e.item_id].format == "course"
        ]
        if not started:
            return None
        item = self._items[started[0].item_id]
        return Scored(
            item,
            score=1.0,
            contributions={"history": 1.0},
            params={"history": {"title": item.title}},
            continuing=True,
        )

    # Special answers ----------------------------------------------------------------------

    def _low_confidence(self, context: _Context, understanding: Understanding) -> bool:
        """Unclear free text: no task cues, or several equal tasks with weak closeness."""
        if context.profile.tasks or any(weight > 0 for weight in context.history_weights.values()):
            return False
        text = (context.profile.free_text or "").strip()
        if not text:
            return False
        if not understanding.has_signal:
            return True
        if len(understanding.tasks) < 2:
            return False
        # Closeness of the user's own words, not of the task descriptions added to the query.
        hits = self._index.search(text, limit=1)
        return (hits[0].score if hits else 0.0) < CONFIDENT_SEMANTIC

    def _clarify(
        self,
        profile: Profile,
        context: _Context,
        question: str,
        limit: int,
        scored: list[Scored] | None = None,
    ) -> RecommendationResponse:
        """2-4 task options and a general selection instead of an invented match."""
        counter: Counter[str] = Counter()
        for entry in (scored or [])[:40]:
            for task, strength in self._facts[entry.item.item_id].tasks.items():
                if strength >= 0.8:
                    counter[task] += 1
        options = [self._taxonomy.tasks[task].label for task, _ in counter.most_common(4)]
        for task_id in ("business_start", "marketing", "taxes_reporting", "team_hiring"):
            label = self._taxonomy.tasks[task_id].label
            if len(options) < 4 and label not in options:
                options.append(label)
        general = self._starter(context)[:limit]
        return RecommendationResponse(
            more=[self._to_recommendation(entry, context) for entry in general],
            clarification=Clarification(question=question, options=options[:4]),
            mode="ours",
            profile_used=self._profile_used(profile, context),
        )

    def _starter(self, context: _Context) -> list[Scored]:
        """Cold-start set (spec 6.5): proven course, tax calculator, template, articles, news."""
        free = [i for i in self._items.values() if i.item_id not in context.excluded]

        def best(pool: Iterable[Item], key) -> Item | None:
            ordered = sorted(pool, key=lambda i: (key(i), i.item_id), reverse=True)
            return ordered[0] if ordered else None

        picks: list[Item] = []
        course = best(
            (i for i in free if i.format == "course" and i.access == "free"
             and self._facts[i.item_id].level == "beginner"),
            lambda i: (i.completion_rate or 0) * math.log1p(i.popularity or 0),
        )  # fmt: skip
        calculator = best(
            (i for i in free if i.item_type == "calculator" and "налог" in i.title.casefold()),
            lambda i: i.popularity or 0,
        )
        template = best(
            (i for i in free if i.format == "template"
             and "business_start" in self._facts[i.item_id].tasks),
            lambda i: (i.published_at or self._now).timestamp(),
        )  # fmt: skip
        topic_counts = Counter(t for i in free for t in i.topics)
        articles = sorted(
            (i for i in free if i.item_type == "article" and i.access == "free" and i.topics),
            key=lambda i: (-topic_counts[i.topics[0]], -(i.published_at or self._now).timestamp(),
                           i.item_id),
        )  # fmt: skip
        fresh = best(
            (i for i in free if i.item_type == "article"
             and {match_key(t) for t in i.topics} & LAW_CHANGE_LABELS),
            lambda i: (i.published_at or self._now).timestamp(),
        )  # fmt: skip
        topics_used: set[str] = set()
        for item in [course, calculator, template, fresh]:
            if item and item not in picks:
                picks.append(item)
        for article in articles:
            if len(picks) >= STARTER_SIZE:
                break
            if article.topics[0] not in topics_used and article not in picks:
                picks.append(article)
                topics_used.add(article.topics[0])
        result = []
        for item in picks:
            contributions = {"popularity": 1.0} if item.popularity else {"quality": 0.5}
            if item.format in {"quick_answer", "tool"}:
                contributions["format"] = 0.5
            result.append(Scored(item, 1.0, contributions, {}))
        return result

    def _cold_start(
        self, profile: Profile, context: _Context, limit: int
    ) -> RecommendationResponse:
        starter = self._starter(context)
        used = {entry.item.item_id for entry in starter}
        plan = self._week_plan(context, used)
        used |= {entry.item.item_id for entry in plan}
        quick = [
            Scored(i, 0.5, {"format": 0.5}, {})
            for i in sorted(self._items.values(), key=lambda i: (-(i.popularity or 0), i.item_id))
            if i.item_id not in used | context.excluded and self._is_quick(i)
        ][:QUICK_LIMIT]
        return RecommendationResponse(
            route=[self._to_recommendation(entry, context) for entry in plan],
            more=[self._to_recommendation(entry, context) for entry in starter[:limit]],
            quick=[self._to_recommendation(entry, context) for entry in quick],
            mode="ours",
            profile_used=self._profile_used(profile, context),
        )

    def _week_plan(self, context: _Context, used: set[str]) -> list[Scored]:
        """'Plan for the week': five short materials on one topic, one per day."""
        short = [
            i
            for i in self._items.values()
            if i.item_id not in used | context.excluded
            and i.item_type == "article"
            and i.access == "free"
            and i.duration_min is not None
            and i.duration_min <= WEEK_PLAN_MAX_MIN
            # One topic for the whole week: materials filed under starting a business.
            and {match_key(t) for t in i.topics} & START_LABELS
        ]
        short.sort(key=lambda i: (LEVELS.index(self._facts[i.item_id].level),
                                  i.duration_min or 0, i.item_id))  # fmt: skip
        return [Scored(i, 0.5, {"format": 0.5}, {}) for i in short[:WEEK_PLAN_DAYS]]

    # Output ----------------------------------------------------------------------------

    def _profile_used(self, profile: Profile, context: _Context) -> Profile:
        """The profile the answer used, including what was understood from text or history."""
        return profile.model_copy(
            update={"tasks": context.tasks, "stage": context.stage, "experience": context.level},
            deep=True,
        )

    def _to_recommendation(self, entry: Scored, context: _Context) -> Recommendation:
        reasons: list[Reason] = []
        item = entry.item
        facts = self._facts[item.item_id]
        for name, value in sorted(entry.contributions.items(), key=lambda kv: (-kv[1], kv[0])):
            if value <= 0:
                continue
            reason = self._reason(name, value, entry, facts, context)
            if reason and reason.code not in {r.code for r in reasons}:
                reasons.append(reason)
        # Facts about the material itself complete a short list; none contradicts the profile.
        for reason in self._fact_reasons(item, facts, context):
            if len(reasons) >= 3:
                break
            if reason.code not in {r.code for r in reasons}:
                reasons.append(reason)
        step = next_step_for(item)
        if entry.continuing:
            step = NextStep(kind="start_course", label="Продолжить курс", url=item.url)
        return Recommendation(
            item=item.model_copy(deep=True),
            score=round(entry.score, 6),
            reasons=reasons,
            next_step=step,
        )

    def _fact_reasons(self, item: Item, facts: ItemFacts, context: _Context) -> list[Reason]:
        result = []
        if (item.popularity or 0) > 0:
            result.append(Reason(code=ReasonCode.POPULAR_WITH_SIMILAR, weight=0.0))
        if facts.level == "beginner" and context.level in (None, "beginner"):
            result.append(Reason(code=ReasonCode.BEGINNER_FRIENDLY, weight=0.0))
        if item.duration_min and item.duration_min <= FACT_MAX_MIN:
            result.append(
                Reason(code=ReasonCode.SHORT_FORMAT, weight=0.0,
                       params={"minutes": item.duration_min})
            )  # fmt: skip
        elif item.format == "tool":
            # A calculator answers at once; its time is not recorded in the data.
            result.append(Reason(code=ReasonCode.SHORT_FORMAT, weight=0.0))
        elif item.format == "template":
            result.append(
                Reason(code=ReasonCode.SHORT_FORMAT, weight=0.0, params={"format": "template"})
            )
        labels = {match_key(v) for v in [*item.topics, *(filter(None, [item.subject]))]}
        recent = item.published_at and (self._now - item.published_at) <= timedelta(days=FRESH_DAYS)
        if recent and labels & LAW_CHANGE_LABELS:
            result.append(Reason(code=ReasonCode.FRESH, weight=0.0))
        if item.format == "course" and item.access == "free":
            result.append(Reason(code=ReasonCode.DEEP_DIVE_COURSE, weight=0.0))
        # Topic-level links (0.5) are enough to name the material's subject truthfully.
        own = [(s, t) for t, s in facts.tasks.items() if s >= 0.5]
        if own:
            # Last resort, worded as the material's topic rather than the user's task.
            task = max(own)[1]
            result.append(
                Reason(code=ReasonCode.TASK_MATCH, weight=0.0, params={"task": task, "about": True})
            )
        return result

    def _reason(
        self, name: str, value: float, entry: Scored, facts: ItemFacts, context: _Context
    ) -> Reason | None:
        weight = round(value, 6)
        params = entry.params.get(name, {})
        if name == "task" and params.get("task"):
            return Reason(code=ReasonCode.TASK_MATCH, weight=weight, params=params)
        if name == "stage" and facts.stages and value >= 0.5 * self._weights["stage"]:
            return Reason(code=ReasonCode.STAGE_FIT, weight=weight, params=params)
        if name == "level":
            if context.level_raised and facts.level != "beginner":
                return Reason(code=ReasonCode.NEXT_LEVEL, weight=weight)
            if facts.level == "beginner" and context.level == "beginner":
                return Reason(code=ReasonCode.BEGINNER_FRIENDLY, weight=weight)
            return None
        if name == "industry" and params:
            return Reason(code=ReasonCode.INDUSTRY_MATCH, weight=weight, params=params)
        if name == "history" and params.get("title"):
            return Reason(
                code=ReasonCode.CONTINUES_HISTORY, weight=weight, params={"title": params["title"]}
            )
        if name in {"popularity", "quality", "sequence"} and (entry.item.popularity or 0) > 0:
            return Reason(code=ReasonCode.POPULAR_WITH_SIMILAR, weight=weight)
        if name == "format" and entry.item.duration_min and entry.item.duration_min <= 10:
            return Reason(
                code=ReasonCode.SHORT_FORMAT, weight=weight,
                params={"minutes": entry.item.duration_min},
            )  # fmt: skip
        if name == "fresh":
            return Reason(code=ReasonCode.FRESH, weight=weight)
        if name == "offer" or (name == "format" and entry.item.format == "course"):
            return Reason(code=ReasonCode.DEEP_DIVE_COURSE, weight=weight)
        return None


DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "retrieval_tfidf"


def load_ranker(
    catalog_path: Path | None = None,
    index_path: Path = DEFAULT_INDEX_PATH,
    weights: Mapping[str, float] | None = None,
    base_url: str | None = None,
    transitions_path: Path | None = DEFAULT_TRANSITIONS_PATH,
    transitions: CourseTransitions | None = None,
    rerank: bool = True,
    encoder_mode: str | None = None,
    ml_path: Path | None = None,
    model_path: Path | None = None,
    weights_path: Path | None = DEFAULT_WEIGHTS_PATH,
) -> Ranker:
    """Ranker over the committed catalog and the B04 fallback index, with full index checks."""
    from alfarec.pipeline.build_index import prepare_texts
    from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog, with_base_url
    from alfarec.recsys.retrieval import digest, texts_digest
    from alfarec.recsys.taxonomy import load_taxonomy

    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    saved_weights = {}
    if weights_path is not None:
        configured = (
            os.environ.get("WEIGHTS_PATH") if weights_path == DEFAULT_WEIGHTS_PATH else None
        )
        path = Path(configured) if configured else weights_path
        if path.exists() or configured or path != DEFAULT_WEIGHTS_PATH:
            saved_weights = load_weights(path, catalog_path)
    effective_weights = {
        **saved_weights,
        **validate_weights(weights if weights is not None else {}),
    }
    catalog = load_catalog(catalog_path)
    ids, texts = prepare_texts(catalog, source="metadata")
    index = TfidfIndex.load(
        index_path,
        expected_item_ids=ids,
        expected_catalog_sha256=digest(catalog_path),
        expected_text_source="metadata",
        expected_texts_sha256=texts_digest(ids, texts),
    )
    from alfarec.recsys.ml_artifacts import (
        DEFAULT_ML_PATH,
        checked_manifest,
        load_dense,
        load_graph,
        load_labels,
    )

    encoder_mode = encoder_mode or os.environ.get("ENCODER", "auto")
    if encoder_mode not in {"auto", "tfidf", "dense"}:
        raise ValueError("ENCODER must be auto, tfidf or dense")
    ml_path = ml_path or Path(os.environ.get("ML_PATH", str(DEFAULT_ML_PATH)))
    labels = graph = None
    taxonomy = load_taxonomy()
    if encoder_mode != "tfidf" and (ml_path / "manifest.json").exists():
        manifest = checked_manifest(ml_path, catalog_path, ids)
        model_path = model_path or Path(os.environ.get("MODEL_PATH", str(ml_path / "model")))
        try:
            dense = load_dense(ml_path, manifest, model_path)
        except ImportError:
            dense = None
        if dense is not None:
            index = dense
            labels = load_labels(ml_path / "item_labels.parquet", ids, taxonomy)
            # Full-snapshot co-enrolments are packaged separately. Use only semantic edges
            # here so hidden-next-course evaluation cannot observe future enrolments.
            graph = load_graph(ml_path / "semantic_graph.npz", len(ids))
        elif encoder_mode == "dense":
            raise ValueError("Requested dense encoder weights/dependencies unavailable")
    elif encoder_mode == "dense":
        raise ValueError("Requested dense artifacts unavailable")
    if base_url:
        catalog = with_base_url(catalog, base_url)
    if transitions is None and transitions_path is not None and transitions_path.exists():
        # Without the file the ranker still works, on content and popularity alone.
        transitions = CourseTransitions.load(transitions_path)
    ranker = Ranker(
        catalog, taxonomy, index, weights=effective_weights, transitions=transitions, rerank=rerank
    )
    if labels is not None:
        for item_id, row in labels.items():
            old = ranker._facts[item_id]
            ranker._facts[item_id] = ItemFacts(
                {task: 1.0 for task in row["tasks"]},
                frozenset(row["stages"]),
                row["level"] or old.level,
                frozenset(row["industries"]),
                old.topic,
            )
        ranker._graph = graph
        ranker._graph_ids = ids
        ranker._graph_positions = {item: n for n, item in enumerate(ids)}
    return ranker
