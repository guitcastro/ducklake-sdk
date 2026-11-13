import os
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ducklake as dl

pytestmark = pytest.mark.skip_config(
    storage="s3", reason="Local Parquet files are written to the storage path."
)


def _write_parquet(table: dl.Table, name: str, data: dict) -> str:
    _, generator, _, _ = table._get_write_info()
    path = os.path.join(generator.base_path, name)
    pl.DataFrame(data).write_parquet(
        path,
        arrow_schema=table.schema,
        storage_options=table._storage_options.to_dict(),
        mkdir=True,
    )
    return name


def test_write_data_files_from_paths(ducklake: dl.Ducklake, random_table_name: str) -> None:
    # Arrange
    table = ducklake.create_table(random_table_name, {"x": dl.Int64()})
    path = _write_parquet(table, "file.parquet", {"x": [1, 2, 3]})

    # Act
    table.write_data_files([path])

    # Assert
    assert table.scan().data_files[0].statistics.num_rows == 3
    assert_frame_equal(
        table.scan_polars().sort("x"),
        pl.LazyFrame({"x": [1, 2, 3]}, schema={"x": pl.Int64}),
    )


def test_write_data_files_with_statistics(ducklake: dl.Ducklake, random_table_name: str) -> None:
    # Arrange
    table = ducklake.create_table(random_table_name, {"x": dl.Int64()})
    path = _write_parquet(table, "file.parquet", {"x": [1, 2, 3]})

    # Act
    table.write_data_files(
        [
            dl.WriteDataFile(
                path,
                statistics=dl.DataFileStatistics(
                    num_rows=3,
                    column_stats={1: dl.ColumnStats(min_value=1, max_value=3, null_count=0)},
                ),
            )
        ]
    )

    # Assert
    stats = table.scan().data_files[0].statistics
    assert stats.num_rows == 3
    assert stats.column_stats[1].min_value == 1
    assert stats.column_stats[1].max_value == 3


def test_write_data_files_mixed(ducklake: dl.Ducklake, random_table_name: str) -> None:
    # Arrange
    table = ducklake.create_table(random_table_name, {"x": dl.Int64()})
    path_a = _write_parquet(table, "a.parquet", {"x": [1, 2, 3]})
    path_b = _write_parquet(table, "b.parquet", {"x": [4, 5, 6]})

    # Act
    table.write_data_files([path_a, dl.WriteDataFile(path_b)])

    # Assert
    assert_frame_equal(
        table.scan_polars().sort("x"),
        pl.LazyFrame({"x": [1, 2, 3, 4, 5, 6]}, schema={"x": pl.Int64}),
    )


def test_write_data_files_in_transaction(ducklake: dl.Ducklake, random_table_name: str) -> None:
    # Arrange
    table = ducklake.create_table(random_table_name, {"x": dl.Int64()})
    path = _write_parquet(table, "file.parquet", {"x": [1, 2, 3]})

    # Act
    with ducklake.transaction() as tx:
        tx.table(random_table_name).write_data_files([path])

    # Assert
    assert_frame_equal(
        ducklake.table(random_table_name).scan_polars().sort("x"),
        pl.LazyFrame({"x": [1, 2, 3]}, schema={"x": pl.Int64}),
    )


@pytest.mark.parametrize("city_column", ["city", "CITY"])
def test_external_parquet_columns_with_different_field_ids_are_mapped_by_name(
    ducklake: dl.Ducklake, random_table_name: str, tmp_path: Path, city_column: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Arrange
    table = ducklake.create_table(
        random_table_name, {"name": dl.Varchar(), "city": dl.Varchar(), "country": dl.Varchar()}
    )
    schema = pa.schema(
        [
            pa.field(city_column, pa.string(), metadata={b"PARQUET:field_id": b"1"}),
            pa.field("name", pa.string(), metadata={b"PARQUET:field_id": b"2"}),
        ]
    )
    data = pa.Table.from_pydict(
        {city_column: ["Recife", "Belém"], "name": ["Ana", "Bia"]}, schema=schema
    )
    path = str(tmp_path / "external.parquet")
    pq.write_table(data, path)

    # Act
    table.write_data_files_by_name([path])

    # Assert
    mapping = table.scan().data_files[0].name_mapping
    assert {entry["source_name"]: entry["target_field_id"] for entry in mapping} == {
        "name": 1,
        "city": 2,
        "country": 3,
    }
    assert table._duckdb_connection.execute(
        f"SELECT name, city, country FROM {table.name}"
    ).fetchall() == [("Ana", "Recife", None), ("Bia", "Belém", None)]
    assert table.read_polars().to_dicts() == [
        {"name": "Ana", "city": "Recife", "country": None},
        {"name": "Bia", "city": "Belém", "country": None},
    ]
    table._duckdb_connection.execute(f"DELETE FROM {table.name} WHERE name = 'Ana'")
    assert table.read_polars().to_dicts() == [{"name": "Bia", "city": "Belém", "country": None}]
    table.rename_column("name", "person")
    assert table.read_polars().to_dicts() == [{"person": "Bia", "city": "Belém", "country": None}]


@pytest.mark.parametrize("include_name", [True, False])
def test_external_nested_columns_are_mapped_by_name(
    ducklake: dl.Ducklake, random_table_name: str, tmp_path: Path, include_name: bool
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Arrange
    table = ducklake.create_table(
        random_table_name, {"person": dl.Struct({"name": dl.Varchar(), "city": dl.Varchar()})}
    )
    fields = [pa.field("city", pa.string(), metadata={b"PARQUET:field_id": b"2"})]
    if include_name:
        fields.append(pa.field("name", pa.string(), metadata={b"PARQUET:field_id": b"3"}))
    schema = pa.schema(
        [pa.field("person", pa.struct(fields), metadata={b"PARQUET:field_id": b"1"})]
    )
    path = tmp_path / "nested.parquet"
    pq.write_table(
        pa.Table.from_pydict({"person": [{"city": "Recife", "name": "Ana"}, None]}, schema=schema),
        path,
    )

    # Act
    table.write_data_files_by_name([str(path)])

    # Assert
    expected = [
        {"person": {"name": "Ana" if include_name else None, "city": "Recife"}},
        {"person": None},
    ]
    assert table._duckdb_connection.execute(f"SELECT * FROM {table.name}").fetchall() == [
        (expected[0]["person"],),
        (None,),
    ]
    assert table.read_polars().to_dicts() == expected


def test_name_mapping_rejects_partitioned_tables_until_supported(
    ducklake: dl.Ducklake, random_table_name: str
) -> None:
    # Arrange
    table = ducklake.create_table(random_table_name, {"region": dl.Varchar()}, partition_by="region")
    file = dl.WriteDataFile("data.parquet", statistics=dl.DataFileStatistics(num_rows=1))

    # Act / Assert
    with pytest.raises(RuntimeError, match="partition"):
        table.write_data_files_by_name([file])
