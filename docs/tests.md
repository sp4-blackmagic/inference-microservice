# Microservice tests
```shell
pytest tests/ --doctest-modules --junitxml=junit/test-results.xml --cov=src --cov-report=xml --cov-report=html
```
### Unit tests
Results in `/junit/test-results.xml`

### Integration tests

### Coverage
Results in `/htmlcov/index.html`