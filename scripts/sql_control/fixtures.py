"""Original fixed SQLite sanity fixtures, disjoint from the dataset protocol.

Every query runs on the same 40 populated rows for three fixed seeds. These
small synthetic cases do not constitute a broad SQL execution benchmark.
"""

import random

from scripts.freight import digest

SCHEMA = ('CREATE TABLE control17_staff (id INTEGER, name TEXT, team TEXT, pay INTEGER, '
          'bonus INTEGER); CREATE TABLE control17_teams (team TEXT, city TEXT);')
TASKS = [
    ('List names of staff paid more than 80.', 'SELECT name FROM control17_staff WHERE pay > 80'),
    ('How many staff are in Engineering?', "SELECT COUNT(*) FROM control17_staff WHERE team = 'Engineering'"),
    ('What is the maximum pay?', 'SELECT MAX(pay) FROM control17_staff'),
    ('List teams and their average pay.', 'SELECT team, AVG(pay) FROM control17_staff GROUP BY team'),
    ('List the three highest paid staff names, breaking pay ties by id ascending.',
     'SELECT name FROM control17_staff ORDER BY pay DESC, id ASC LIMIT 3'),
    ('List names of staff with no bonus.', 'SELECT name FROM control17_staff WHERE bonus IS NULL'),
    ('List different teams.', 'SELECT DISTINCT team FROM control17_staff'),
    ('List names beginning with A.', "SELECT name FROM control17_staff WHERE name LIKE 'A%'"),
    ('List names with pay between 40 and 80 inclusive.',
     'SELECT name FROM control17_staff WHERE pay BETWEEN 40 AND 80'),
    ('List staff names working in Tokyo.',
     "SELECT s.name FROM control17_staff AS s JOIN control17_teams AS t ON s.team = t.team WHERE t.city = 'Tokyo'"),
    ('List teams with more than five staff.',
     'SELECT team FROM control17_staff GROUP BY team HAVING COUNT(*) > 5'),
    ('List names of staff paid above the overall average.',
     'SELECT name FROM control17_staff WHERE pay > (SELECT AVG(pay) FROM control17_staff)'),
]
SEEDS = (1701, 1702, 1703)


def cases():
    return [dict(id=f'fixture-{i:02}', context=SCHEMA, question=q, answer=a,
                 ordered=i == 4) for i, (q, a) in enumerate(TASKS)]


def population(seed):
    rng = random.Random(seed)
    staff = [(i, ('Alice', 'alice', 'Bob', 'Amir', 'Zoe')[i % 5] + str(i),
              ('Engineering', 'engineering', 'Sales', 'HR')[i % 4],
              rng.choice((20, 40, 60, 80, 100, 120)),
              None if i % 3 == 0 else rng.randrange(1, 10)) for i in range(40)]
    teams = [('Engineering', 'Tokyo'), ('engineering', 'Paris'), ('Sales', 'Tokyo'), ('HR', 'Oslo')]
    return {'staff': staff, 'teams': teams}


def fingerprint():
    return digest({'cases': cases(), 'seeds': SEEDS,
                   'populations': [population(seed) for seed in SEEDS]})
