#!/usr/bin/env python3
"""Consistent copy of memory.db into ~/backups/memory-core; keeps the newest 14."""
import glob, os, sqlite3, time

from config import DB_PATH

SRC = str(DB_PATH)
DST = os.path.expanduser(os.getenv("MEMORY_BACKUP_DIR", "~/backups/memory-core"))
os.makedirs(DST, mode=0o700, exist_ok=True)
out = f"{DST}/memory-{time.strftime('%Y%m%d-%H%M%S')}.db"
s, d = sqlite3.connect(SRC), sqlite3.connect(out)
s.backup(d); d.close(); s.close()
os.chmod(out, 0o600)
for f in sorted(glob.glob(f"{DST}/memory-*.db"))[:-14]:
    os.remove(f)
print("backup ok:", out)
