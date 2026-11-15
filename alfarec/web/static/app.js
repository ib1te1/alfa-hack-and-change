"use strict";

const TYPE_LABELS = {
  article: "Статья", document: "Документ", course: "Курс", webinar: "Вебинар",
  podcast: "Подкаст", glossary: "Глоссарий", calculator: "Калькулятор",
};
const REASON_TEXTS = {
  task_match: (p) => (p.task ? `Подходит под задачу «${p.task}»` : "Подходит под вашу задачу"),
  stage_fit: () => "Подходит для вашего этапа",
  beginner_friendly: () => "Для тех, кто только начинает",
  industry_match: (p) => (p.industry ? `Для сферы «${p.industry}»` : "Для вашей сферы"),
  continues_history: (p) => (p.title ? `Продолжает «${p.title}»` : "Продолжает то, что вы уже изучили"),
  popular_with_similar: () => "Популярно у похожих предпринимателей",
  short_format: (p) => (p.minutes ? `Займёт ${p.minutes} мин` : "Короткий формат"),
  fresh: () => "Свежий материал",
  next_level: () => "Следующий шаг по уровню",
  deep_dive_course: () => "Системный курс по теме",
};
const STORAGE = { user: "alfarec.user", skip: "alfarec.onboarding-skipped" };

const state = {
  meta: null,
  personas: [],
  profile: emptyProfile(),
  userId: null,
  compare: false,
  requestSeq: 0,
};

function emptyProfile() {
  return { stage: null, industries: [], tasks: [], experience: null, time_budget: null, free_text: null };
}

function storageGet(key) {
  try { return window.localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch { /* storage may be unavailable */ }
}

function anonymousUser() {
  let id = storageGet(STORAGE.user);
  if (!id) {
    id = "web-" + Math.random().toString(36).slice(2, 12);
    storageSet(STORAGE.user, id);
  }
  return id;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `Ошибка ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch { /* keep the status message */ }
    throw new Error(message);
  }
  return response.json();
}

function notice(text) {
  const node = document.getElementById("notice");
  node.textContent = text || "";
  node.hidden = !text;
}

function renderChips() {
  const sources = {
    stage: state.meta.stages, tasks: state.meta.tasks, industries: state.meta.industries,
    experience: state.meta.levels, time_budget: state.meta.time_budgets,
  };
  for (const box of document.querySelectorAll(".panel .chips[data-field]")) {
    const field = box.dataset.field;
    const single = box.hasAttribute("data-single");
    box.replaceChildren();
    for (const option of sources[field]) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = option.label;
      if (option.description) chip.title = option.description;
      const selected = single ? state.profile[field] === option.id : state.profile[field].includes(option.id);
      chip.setAttribute("aria-pressed", String(selected));
      chip.addEventListener("click", () => {
        if (single) {
          state.profile[field] = state.profile[field] === option.id ? null : option.id;
        } else {
          const values = state.profile[field];
          state.profile[field] = values.includes(option.id)
            ? values.filter((value) => value !== option.id)
            : [...values, option.id];
        }
        renderChips();
        scheduleRefresh();
      });
      box.append(chip);
    }
  }
  document.getElementById("free-text").value = state.profile.free_text || "";
}

function accessLabel(item) {
  if (item.access === "subscription") return "По подписке Курс+";
  if (item.access === "paid") return item.price_rub ? `${item.price_rub.toLocaleString("ru-RU")} ₽` : "Платно";
  return "Бесплатно";
}

function formatLabel(code) {
  const found = state.meta.formats.find((entry) => entry.id === code);
  return found ? found.label : code;
}

function renderCard(recommendation, interactive) {
  const { item } = recommendation;
  const node = document.getElementById("card-template").content.firstElementChild.cloneNode(true);
  const kind = [TYPE_LABELS[item.item_type] || item.item_type];
  if (formatLabel(item.format) !== kind[0]) kind.push(formatLabel(item.format));
  if (item.duration_min) kind.push(`${item.duration_min} мин`);
  node.querySelector(".card__kind").textContent = kind.join(" · ");
  const access = node.querySelector(".card__access");
  access.textContent = accessLabel(item);
  access.classList.add(`card__access--${item.access}`);
  node.querySelector(".card__title").textContent = item.title;
  node.querySelector(".card__summary").textContent = item.summary;

  const reasons = recommendation.reasons.slice(0, 3)
    .map((reason) => reason.text || (REASON_TEXTS[reason.code] || (() => null))(reason.params || {}))
    .filter(Boolean);
  if (reasons.length) {
    const list = node.querySelector(".card__reasons");
    for (const text of reasons) {
      const li = document.createElement("li");
      li.textContent = text;
      list.append(li);
    }
    node.querySelector(".card__why").hidden = false;
  }

  const next = node.querySelector(".card__next");
  next.textContent = recommendation.next_step.label;
  next.href = recommendation.next_step.url;
  next.addEventListener("click", () => { sendEvent(item.item_id, "view").catch(() => {}); });

  const known = node.querySelector(".card__known");
  const dislike = node.querySelector(".card__dislike");
  if (interactive) {
    known.addEventListener("click", () => feedback(item.item_id, "known", known));
    dislike.addEventListener("click", () => feedback(item.item_id, "dislike", dislike));
  } else {
    known.remove();
    dislike.remove();
  }
  return node;
}

function block(title, recommendations, options = {}) {
  const section = document.createElement("section");
  section.className = "block";
  const heading = document.createElement("h2");
  heading.className = "block__title";
  heading.textContent = title;
  section.append(heading);
  if (options.note) {
    const note = document.createElement("p");
    note.className = "block__note";
    note.textContent = options.note;
    section.append(note);
  }
  const cards = document.createElement("div");
  cards.className = options.route ? "cards route" : "cards";
  for (const recommendation of recommendations) cards.append(renderCard(recommendation, options.interactive));
  section.append(cards);
  return section;
}

function clarification(data) {
  const box = document.createElement("section");
  box.className = "clarify";
  const heading = document.createElement("h2");
  heading.textContent = data.question;
  const chips = document.createElement("div");
  chips.className = "chips";
  for (const option of data.options) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = option;
    chip.addEventListener("click", () => {
      state.profile.free_text = option;
      renderChips();
      refresh();
    });
    chips.append(chip);
  }
  box.append(heading, chips);
  return box;
}

function renderColumn(title, response, interactive) {
  const column = document.createElement("div");
  column.className = "column";
  if (title) {
    const heading = document.createElement("div");
    heading.className = "column__title";
    heading.textContent = title;
    column.append(heading);
  }
  if (response.clarification) column.append(clarification(response.clarification));
  if (response.route.length) {
    // With an empty profile the ranker puts the week plan in the route: one short material a day.
    const emptyProfile = !Object.values(response.profile_used || {}).some((value) => (Array.isArray(value) ? value.length : value));
    const title = emptyProfile ? "План на неделю" : "Ваш маршрут";
    const note = emptyProfile ? "По одному короткому материалу в день — чтобы вернуться завтра" : response.route_note;
    column.append(block(title, response.route, { route: true, interactive, note }));
  }
  if (response.more.length) column.append(block("Ещё по теме", response.more, { interactive }));
  if (response.quick.length) {
    column.append(block("Быстро, за 5 минут", response.quick, { interactive, note: "Короткие материалы, чтобы вернуться завтра" }));
  }
  if (!response.route.length && !response.more.length && !response.quick.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Подходящих материалов не нашлось. Попробуйте убрать часть условий.";
    column.append(empty);
  }
  return column;
}

function requestBody(mode) {
  const profile = { ...state.profile };
  for (const key of Object.keys(profile)) {
    if (profile[key] === null || (Array.isArray(profile[key]) && !profile[key].length)) delete profile[key];
  }
  return JSON.stringify({ profile, user_id: state.userId, mode, limit: 10 });
}

let refreshTimer = null;

// Every answer updates the results at once; a short pause folds a burst of clicks into one request.
function scheduleRefresh() {
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => {
    document.getElementById("onboarding").hidden = true;
    refresh();
  }, 250);
}

async function refresh() {
  const columns = document.getElementById("columns");
  columns.setAttribute("aria-busy", "true");
  const seq = ++state.requestSeq;
  try {
    const modes = state.compare ? ["ours", "baseline"] : ["ours"];
    const responses = await Promise.all(
      modes.map((mode) => api("/api/recommendations", { method: "POST", body: requestBody(mode) })),
    );
    // A slower earlier request must not overwrite the answer to a newer one.
    if (seq !== state.requestSeq) return;
    columns.className = state.compare ? "columns columns--compare" : "columns";
    columns.replaceChildren(
      ...responses.map((response, index) => renderColumn(
        state.compare ? (modes[index] === "ours" ? "Наша выдача" : "Бейзлайн: по теме и популярности") : null,
        response,
        modes[index] === "ours",
      )),
    );
    notice("");
  } catch (error) {
    notice(`Не удалось получить подборку: ${error.message}`);
  } finally {
    columns.removeAttribute("aria-busy");
  }
}

function sendEvent(itemId, type) {
  return api("/api/events", {
    method: "POST",
    body: JSON.stringify({ user_id: state.userId, item_id: itemId, type, ts: new Date().toISOString() }),
  });
}

async function feedback(itemId, type, button) {
  button.disabled = true;
  try {
    await sendEvent(itemId, type);
    await refresh();
  } catch (error) {
    notice(`Не удалось сохранить отметку: ${error.message}`);
    button.disabled = false;
  }
}

function renderPersonas() {
  if (!state.personas.length) return;
  const select = document.getElementById("persona");
  const own = new Option("Мой профиль", "");
  select.append(own, ...state.personas.map((persona) => new Option(persona.title, persona.id)));
  select.addEventListener("change", () => {
    const persona = state.personas.find((entry) => entry.id === select.value);
    state.profile = persona ? { ...emptyProfile(), ...structuredClone(persona.profile) } : emptyProfile();
    state.userId = persona ? persona.user_id : anonymousUser();
    const legend = document.getElementById("legend");
    legend.textContent = persona ? `${persona.title}. ${persona.legend}` : "";
    legend.hidden = !persona;
    renderChips();
    refresh();
  });
  document.getElementById("persona-wrap").hidden = false;
}

function setupOnboarding() {
  const card = document.getElementById("onboarding");
  card.hidden = storageGet(STORAGE.skip) === "1";
  document.getElementById("onboarding-skip").addEventListener("click", () => {
    card.hidden = true;
    storageSet(STORAGE.skip, "1");
  });
  document.getElementById("onboarding-start").addEventListener("click", () => {
    card.hidden = true;
    const first = document.querySelector(".panel .chip");
    if (!first) return;
    first.scrollIntoView({ behavior: "smooth", block: "center" });
    first.focus({ preventScroll: true });
  });
}

async function init() {
  state.userId = anonymousUser();
  setupOnboarding();
  try {
    [state.meta, state.personas] = await Promise.all([api("/api/meta"), api("/api/personas")]);
  } catch (error) {
    notice(`Сервис недоступен: ${error.message}`);
    return;
  }
  renderChips();
  renderPersonas();

  document.getElementById("profile-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = document.getElementById("free-text").value.trim();
    state.profile.free_text = text || null;
    document.getElementById("onboarding").hidden = true;
    refresh().then(() => {
      if (window.matchMedia("(max-width: 900px)").matches) {
        document.getElementById("results-root").scrollIntoView({ behavior: "smooth" });
      }
    });
  });
  document.getElementById("reset").addEventListener("click", () => {
    state.profile = emptyProfile();
    renderChips();
    refresh();
  });
  document.getElementById("compare").addEventListener("change", (event) => {
    state.compare = event.target.checked;
    refresh();
  });
  await refresh();
}

document.addEventListener("DOMContentLoaded", init);
