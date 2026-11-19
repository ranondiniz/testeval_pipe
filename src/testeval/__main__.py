import sys

from .cli import main

# console do Windows (cp1252) não imprime "→", "†", "✓": força UTF-8 com substituição
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

main()
