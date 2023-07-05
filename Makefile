SHELL = /bin/bash

INP_GRAPH_DIR = input_graph_CSV_files

TEST_DATA_LST = $(INP_GRAPH_DIR)/testing_data_files_prefixes.txt
TRAIN_DATA_LST = $(INP_GRAPH_DIR)/training_data_files_prefixes.txt
VALIDATION_DATA_LST = $(INP_GRAPH_DIR)/validation_data_files_prefixes.txt

MODEL_DIR = output_saved_trained_models
RAW_TRAIN_DATA_DIR = training_data

VENV ?= venv

train: $(MODEL_DIR)

$(MODEL_DIR):
	${VENV}/bin/python3 run_training.py

EXAMPLE ?= $(shell head -n1 $(VALIDATION_DATA_LST))

example:
	${VENV}/bin/python3 run_inference_for_one_graph.py ${MODEL_DIR}/epoch25.pth ${INP_GRAPH_DIR}/${EXAMPLE}_vertices_in.csv ${INP_GRAPH_DIR}/${EXAMPLE}_edges.csv /dev/stdout \
		| paste <(tail -n +2 ${INP_GRAPH_DIR}/${EXAMPLE}_vertices_out.csv) -

cleanAll distclean:
	find ${INP_GRAPH_DIR}/data/ -type d | xargs rm -rf
	rm -f ${TEST_DATA_LST} ${TRAIN_DATA_LST} ${VALIDATION_DATA_LST}
	rm -rf ${MODEL_DIR} ${RAW_TRAIN_DATA_DIR}
	rm -rf __pycache__