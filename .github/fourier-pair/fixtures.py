"""Download two attributed prompts to the ephemeral runner. See CMU-ARCTIC-LICENSE.txt."""
from tempfile import TemporaryDirectory
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/MikhailT/cmu-arctic/resolve/b88ad416e092c4dfdffc6071f4b8e2d057a160f9/data/"
FIXTURES = (
    ("a", "bdl-00000-of-00001-0c683a89629915f1.parquet", "bdl-a0207.wav", "4ce6ad7ca7288cb0876071314a9579eb84c8bbcaab7bda558dc65da813a2cff7"),
    ("b", "clb-00000-of-00001-8f46506de584c7ae.parquet", "clb-b0260.wav", "b9f03053810eaac6509dff49f5e32d0255ce8d08e822412cd15dc8df96e115c4"),
)
with TemporaryDirectory() as dest:
    for label, archive, filename, checksum in FIXTURES:
        path, _ = urlretrieve(BASE + archive, Path(dest)/archive)
        for record in pq.ParquetFile(path).iter_batches(batch_size=20):
            for row in record.to_pylist():
                if row["file"] == filename:
                    data = row["audio"]["bytes"]
                    assert hashlib.sha256(data).hexdigest() == checksum
                    Path(label+".wav").write_bytes(data)
                    print(label, filename, row["text"], checksum)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(filename)
