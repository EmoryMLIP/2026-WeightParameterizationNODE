"""Provenance stamp identifying the code state that produced an artifact.

Every generated table/figure/CSV carries this so any number in the paper can be
traced back to a commit.
"""
import datetime
import platform
import subprocess
import sys


def provenance():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        # Scoped to source: generated artifacts (tables/, figures/, results/)
        # are excluded, since writing one necessarily dirties the tree.
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--', 'matlab', 'python', 'data', 'Makefile'],
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        commit, dirty = 'unknown', True
    return dict(commit=commit, dirty=dirty,
                when=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                python=sys.version.split()[0], host=platform.node())


def provenance_line(prefix='%'):
    p = provenance()
    flag = ' (WORKING TREE DIRTY)' if p['dirty'] else ''
    return (f"{prefix} generated {p['when']} from commit {p['commit']}{flag} "
            f"by Python {p['python']} on {p['host']}")
