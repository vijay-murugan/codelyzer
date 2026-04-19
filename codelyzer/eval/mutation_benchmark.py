"""
Optional mutation-testing workflow to compare kill rate with vs without LLM-generated tests.

Install: ``pip install mutmut`` (or Cosmic Ray for larger projects).

This module does not invoke mutmut from Codelyzer; use the shell workflow below on a
small benchmark checkout so runs finish in reasonable time.
"""

MUTATION_EVAL_README = """
Mutation benchmark (optional)

1. Clone or copy a small Python benchmark repo with an existing test suite.
2. Record baseline: run mutmut with only human-written tests; note survived/killed.
3. Run codelyzer analyze ... --generate-tests (and optionally --auto-validate-tests)
   on a branch/commit so new tests exist.
4. Run mutmut again limiting paths to changed modules; compare kill rate.

Example (mutmut):

  cd /path/to/benchmark_repo
  mutmut run --paths-to-mutate=src/mypkg --tests-dir=tests
  mutmut results

Compare ``survived`` counts before vs after adding generated tests. Lower survived
with the same mutant set means the generated tests improved fault detection.

For CI, pin Python version and mutant seed where the tool supports it so results
are reproducible.
""".strip()
