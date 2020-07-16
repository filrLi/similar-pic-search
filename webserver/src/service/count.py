import logging

from common.config import DEFAULT_TABLE
from common.const import default_cache_dir
from diskcache import Cache
from encoder.encode import feature_extract
from indexer.index import (count_table, create_index, create_table,
                           delete_table, insert_vectors, milvus_client,
                           search_vectors)
from preprocessor.xception import XceptNet


def do_count(table_name):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        index_client = milvus_client()
        print("get table rows:", table_name)
        num = count_table(index_client, table_name=table_name)
        return num
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)
