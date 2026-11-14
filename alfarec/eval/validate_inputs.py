import json
from collections import Counter

from alfarec.eval.profiles import load_profiles
from alfarec.recsys.taxonomy import load_taxonomy


def main() -> None:
    taxonomy = load_taxonomy()
    suite = load_profiles(taxonomy=taxonomy)
    print(
        json.dumps(
            {
                "profiles": len(suite.profiles),
                "cases": dict(Counter(case.case for case in suite.profiles)),
                "split": dict(Counter(case.split for case in suite.profiles)),
                "tasks": len(taxonomy.tasks),
                "industries": len(taxonomy.industries),
                "source_links_status": taxonomy.source_links_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
