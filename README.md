# T-CISM
Compresión de imágenes satelitales multiespectrales del satélite GOES-East.

## Desarrollo

Requiere [uv](https://docs.astral.sh/uv/) y Python ≥ 3.12.

```bash
uv sync                          # instala dependencias (incluye las de desarrollo)
uv run pre-commit install        # activa el hook pre-commit (una vez por clon)
```

El hook ejecuta ruff (lint + format) y mypy en cada commit. Los mismos checks corren en CI
en cada push y pull request.

Checks manuales:

```bash
uv run ruff check .              # lint (estilo, docstrings, anotaciones, seguridad, ...)
uv run ruff format --check .     # formato (líneas de hasta 120 caracteres)
uv run mypy tests            # tipos (strict)
uv run pytest                    # tests
```
