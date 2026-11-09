.PHONY: check smoke

UV = uv run --isolated --no-project --with-requirements requirements.txt

# Offline gate: every module compiles and every test that needs no embedding server passes.
check:
	$(UV) python -m py_compile *.py
	$(UV) --with pytest --with pytest-asyncio python -m pytest -q tests
	$(UV) python test_supersede.py
	$(UV) python test_skills.py
	$(UV) python test_delta_pack.py

# Needs tokens.json and a live embedding server at EMBEDDING_API_URL.
smoke:
	$(UV) python test_memory.py
