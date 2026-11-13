use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};

use super::DataFileStatistics;

/// Result of scanning a table at a specific snapshot.
pub struct ScanResult {
    /// The data files that need to be read to produce the table contents.
    pub data_files: Vec<ScanDataFile>,
    /// Record batches whose contents are inlined in the catalog.
    pub inline_data: Vec<RecordBatch>,
}

/// A data file that needs to be read as part of a [`ScanResult`].
pub struct ScanDataFile {
    /// Column mapping used to interpret this file, if present in the catalog.
    pub mapping_id: Option<i64>,
    /// Name mapping entries associated with the data file.
    pub name_mapping: Vec<ScanNameMapping>,
    /// The path of the data file.
    pub path: String,
    /// The encryption key stored in the catalog, or `None` for an unencrypted file.
    pub encryption_key: Option<String>,
    /// Statistics describing the contents of the data file.
    pub statistics: DataFileStatistics,
    /// Delete files that need to be applied to the data file.
    pub delete_files: Vec<ScanDeleteFile>,
    /// Row indices that have been deleted via inline deletes.
    pub inline_deletes: Option<Arc<Int64Array>>,
}

/// One field in a data-file name mapping.
#[derive(Debug, Clone)]
pub struct ScanNameMapping {
    pub column_id: i64,
    pub source_name: Option<String>,
    pub target_field_id: Option<i64>,
    pub parent_column: Option<i64>,
    pub is_partition: Option<bool>,
}

/// A delete file that needs to be applied to a data file.
pub struct ScanDeleteFile {
    /// The path of the delete file.
    pub path: String,
    /// The encryption key stored in the catalog, or `None` for an unencrypted file.
    pub encryption_key: Option<String>,
    /// The number of rows deleted by the delete file.
    pub num_deletes: usize,
    /// The size of the delete file in bytes.
    pub file_size_bytes: Option<usize>,
    /// The size of the delete file's footer in bytes.
    pub footer_size_bytes: Option<usize>,
}
