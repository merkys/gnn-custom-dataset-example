INP_GRAPH_DIR = input_graph_CSV_files/

TEST_DATA_LST = $(INP_GRAPH_DIR)/testing_data_files_prefixes.txt
TRAIN_DATA_LST = $(INP_GRAPH_DIR)/training_data_files_prefixes.txt
VALIDATION_DATA_LST = $(INP_GRAPH_DIR)/validation_data_files_prefixes.txt

cleanAll distclean:
	find ${INP_GRAPH_DIR}/data/ -type f | xargs rm -rf
	rm -f ${TEST_DATA_LST} ${TRAIN_DATA_LST} ${VALIDATION_DATA_LST}
