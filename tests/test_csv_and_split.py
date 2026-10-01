"""CSV encodings, path remapping and the three-way split."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ct25d.tabular import apply_path_map, parse_path_map, read_table, split_three

TEXT = "id,name,age\nA1,山田 ,64\nA2,佐藤,#N/A\n"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp932"])
def test_read_table_detects_the_encoding(tmp_path, encoding):
    path = tmp_path / "list.csv"
    path.write_bytes(TEXT.replace("\n", "\r\n").encode(encoding))
    df = read_table(path)
    assert list(df.columns) == ["id", "name", "age"]       # no BOM in "id"
    assert df["name"].tolist() == ["山田", "佐藤"]           # stripped
    assert df["age"].isna().tolist() == [False, True]


def test_path_map_rewrites_only_a_leading_prefix():
    df = pd.DataFrame({"img": ["/home/hanaoka/a.nii.gz", "/home/hanaokax/b",
                               "/data/home/hanaoka/c", None],
                       "other": ["/home/hanaoka/z"] * 4})
    out = apply_path_map(df, ["img"], parse_path_map(["/home/hanaoka=/mnt/w"]))
    assert out["img"].tolist()[:3] == ["/mnt/w/a.nii.gz", "/home/hanaokax/b",
                                       "/data/home/hanaoka/c"]
    assert out["other"].tolist() == df["other"].tolist()
    assert df["img"][0] == "/home/hanaoka/a.nii.gz"          # input untouched


def test_path_map_rejects_a_malformed_spec():
    with pytest.raises(ValueError, match="OLD=NEW"):
        parse_path_map(["/home/hanaoka"])


def test_split_three_keeps_groups_together_and_is_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pid": rng.integers(0, 100, 600)})
    tr, va, te = split_three(df, 0.15, 0.15, group_col="pid", seed=3)
    assert len(set(tr) | set(va) | set(te)) == len(df)
    g = [set(df.pid.iloc[i]) for i in (tr, va, te)]
    assert not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2])
    assert 10 <= len(g[2]) <= 20 and 10 <= len(g[1]) <= 20
    again = split_three(df, 0.15, 0.15, group_col="pid", seed=3)
    assert all(np.array_equal(a, b) for a, b in zip((tr, va, te), again))


def test_split_three_from_a_column():
    df = pd.DataFrame({"s": ["train", "val", "test", "Train ", "excluded"]})
    tr, va, te = split_three(df, split_col="s")
    assert tr.tolist() == [0, 3] and va.tolist() == [1] and te.tolist() == [2]
