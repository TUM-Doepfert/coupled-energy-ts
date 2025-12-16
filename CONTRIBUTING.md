# Contributing to Coupled Time Series Paper

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/coupled_ts_paper.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `python validate.py`
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Formatting

We use `black` for code formatting:

```bash
black src/
```

### Linting

We use `ruff` for linting:

```bash
ruff check src/
```

## Testing

Before submitting a PR, run the validation tests:

```bash
python validate.py
```

All tests should pass.

## Adding New Features

### Adding a New Building Type

1. Update `BUILDING_TYPES` in `config/germany_config.py`
2. Add building parameters to `BUILDING_PARAMETERS`
3. Update `BuildingRCCalculator.calculate_rc_values()` if needed
4. Add occupancy schedule to `CoupledTimeSeriesGenerator.OCCUPANCY_SCHEDULES`
5. Add electricity profile to `CoupledTimeSeriesGenerator.BASE_ELECTRICITY`
6. Test with `validate.py`

### Adding a New Location

1. Update `GERMAN_LOCATIONS` in `config/germany_config.py`
2. Ensure latitude and longitude are correct
3. Run `generate_data.py` to verify

### Adding a New Module

1. Create the module in `src/coupled_ts_paper/`
2. Add appropriate docstrings
3. Update `__init__.py` if needed
4. Add tests to `validate.py`
5. Update README.md with usage examples

## Documentation

- Update README.md if you change user-facing functionality
- Update QUICKSTART.md for any quick start changes
- Add inline comments for complex logic
- Use descriptive commit messages

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Keep changes focused and minimal

## Code Review

All pull requests will be reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Security considerations

## Questions?

If you have questions, please open an issue for discussion.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
