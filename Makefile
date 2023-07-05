INP_GRAPH_DIR = input_graph_CSV_files

TEST_DATA_LST = $(INP_GRAPH_DIR)/testing_data_files_prefixes.txt
TRAIN_DATA_LST = $(INP_GRAPH_DIR)/training_data_files_prefixes.txt
VALIDATION_DATA_LST = $(INP_GRAPH_DIR)/validation_data_files_prefixes.txt

MODEL_DIR = output_saved_trained_models
RAW_TRAIN_DATA_DIR = training_data

cleanAll distclean:
	find ${INP_GRAPH_DIR}/data/ -type d | xargs rm -rf
	rm -f ${TEST_DATA_LST} ${TRAIN_DATA_LST} ${VALIDATION_DATA_LST}
	rm -rf ${MODEL_DIR} ${RAW_TRAIN_DATA_DIR}
	rm -rf __pycache__
