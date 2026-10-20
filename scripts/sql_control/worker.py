"""Standard-library-only disposable SQLite process, invoked with -I -S.

Not a general hostile-code sandbox. Only SQLite reads of a controlled in-memory
fixture are authorized; no extensions, user functions, or disk DBs are enabled.
"""

import json
import resource
import sqlite3
import sys


def run(payload):
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    db = sqlite3.connect(':memory:')
    db.executescript(payload['schema'])  # Trusted repository fixture only.
    db.executemany('INSERT INTO control17_staff VALUES (?, ?, ?, ?, ?)', payload['population']['staff'])
    db.executemany('INSERT INTO control17_teams VALUES (?, ?)', payload['population']['teams'])
    db.execute('PRAGMA query_only=ON')
    db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
    db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16384)
    db.setlimit(sqlite3.SQLITE_LIMIT_VDBE_OP, 10000)
    db.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 32)
    db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
    allowed_functions = {'count', 'min', 'max', 'avg', 'sum', 'total', 'abs', 'round',
                         'lower', 'upper', 'length', 'like', 'coalesce', 'ifnull',
                         'substr', 'substring', 'trim', 'ltrim', 'rtrim'}

    def authorize(action, arg1, arg2, database, trigger):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and arg2.lower() in allowed_functions:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    db.set_authorizer(authorize)
    ticks = 0

    def progress():
        nonlocal ticks
        ticks += 1
        return ticks > 1000

    db.set_progress_handler(progress, 100)
    results = []
    for query in (payload['gold'], payload['prediction']):
        ticks = 0
        cursor = db.execute(query)
        rows = cursor.fetchmany(1001)
        if len(rows) > 1000:
            raise ValueError('row limit')
        results.append({'columns': len(cursor.description), 'rows': rows})
    return results


if __name__ == '__main__':
    try:
        response = json.dumps({'results': run(json.loads(sys.stdin.read(131072)))})
        if len(response) > 131072:
            raise ValueError('output limit')
        print(response)
    except (sqlite3.Error, ValueError, MemoryError) as exc:
        print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)[:200]}))
