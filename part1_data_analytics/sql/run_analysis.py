"""Run the Part 1 SQL analysis queries and save results to a text file.

Executes every query in queries.sql (split by '-- @name:' markers) against
data/traffic.db and writes formatted results to sql_analysis_results.txt
so the output can be committed for grading.
"""
import logging
import re
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent
SQL_PATH = SQL_DIR / "queries.sql"
DB_PATH = SQL_DIR.parents[1] / "data" / "traffic.db"
RESULTS_PATH = SQL_DIR / "sql_analysis_results.txt"

COLUMN_WIDTH = 22


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (name, query) pairs using -- @name: markers."""
    blocks = re.split(r"^-- @name:\s*", sql_text, flags=re.MULTILINE)
    named = []
    for block in blocks[1:]:  # first block is the header comment
        name, _, statement = block.partition("\n")
        named.append((name.strip(), statement.strip().rstrip(";")))
    return named


def format_result(name: str, cursor: sqlite3.Cursor) -> str:
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    lines = [f"### {name}", " | ".join(str(c).ljust(COLUMN_WIDTH) for c in columns)]
    lines.append("-" * len(lines[1]))
    for row in rows:
        lines.append(" | ".join(str(v).ljust(COLUMN_WIDTH) for v in row))
    lines.append(f"({len(rows)} rows)\n")
    return "\n".join(lines)


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found at %s - run load_to_sqlite.py first", DB_PATH)
        sys.exit(1)

    sql_text = SQL_PATH.read_text(encoding="utf-8")
    queries = parse_queries(sql_text)
    logger.info("Loaded %d queries from %s", len(queries), SQL_PATH.name)

    results = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for name, statement in queries:
                cursor = conn.execute(statement)
                block = format_result(name, cursor)
                results.append(block)
                logger.info("Executed query: %s", name)
    except sqlite3.Error:
        logger.error("A query failed to execute", exc_info=True)
        sys.exit(1)

    RESULTS_PATH.write_text("\n".join(results), encoding="utf-8")
    logger.info("Results written to %s", RESULTS_PATH)
    # print is used here only to show the results to the user running the script
    print("\n".join(results))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()