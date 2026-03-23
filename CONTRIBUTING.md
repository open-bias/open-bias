# Contributing

## Setup

```bash
git clone https://github.com/open-bias/open-bias.git
cd open-bias
pip install -e ".[dev]"
```

Verify:

```bash
pytest
openbias --help
```

## Making Changes

1. Create a branch from `main`.
2. Make your changes. Follow the conventions in [docs/developing.md](docs/developing.md).
3. Run the checks:

```bash
make lint
make typecheck
make test
```

4. Open a pull request against `main`.

## Code Style

- **Formatter/linter**: ruff. Run `make format` before committing.
- **Type checking**: mypy in strict mode. All function signatures need type hints.
- **Tests**: pytest. New code needs tests. Run `make test-cov` to check coverage.

See [docs/developing.md](docs/developing.md) for naming conventions, file organization rules, and import ordering.

## Adding a Policy Engine

OpenBias's engine system is pluggable. The short version:

1. Create a new directory under `openbias/policy/engines/your_engine/`.
2. Implement the `PolicyEngine` protocol.
3. Register it with `@register_engine("your_engine")`.
4. Add a README in the engine directory.

Full walkthrough with code examples: [docs/developing.md](docs/developing.md#extension-points).

## Pull Requests

- Keep PRs focused. One logical change per PR.
- Write a clear description of what changed and why.
- If the PR adds a new engine or changes the config schema, update the relevant docs in `docs/`.
- All CI checks must pass before merge.

## Reporting Issues

Use [GitHub Issues](https://github.com/open-bias/open-bias/issues). Include:

- What you did (steps to reproduce).
- What you expected.
- What actually happened.
- Python version, OS, and which engine you're using.

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
