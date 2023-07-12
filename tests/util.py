import os
import tempfile
from subprocess import Popen, PIPE
from textwrap import dedent

from generate_test_checks import main


def filecheck(correct: str, module):
    filecheck_path = os.getenv("FILECHECK")
    assert filecheck_path is not None, "no filecheck path in env"
    assert os.path.exists(filecheck_path), "filecheck executable doesn't exist"

    correct = dedent(correct)
    op = dedent(str(module).strip())
    with tempfile.NamedTemporaryFile() as tmp:
        correct_with_checks = main(correct)
        tmp.write(correct_with_checks.encode())
        tmp.flush()
        p = Popen([filecheck_path, tmp.name], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out, err = map(lambda o: o.decode(), p.communicate(input=op.encode()))
        if len(err):
            raise ValueError(err)
