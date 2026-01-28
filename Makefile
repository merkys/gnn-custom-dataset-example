SHELL = /bin/bash

INP_GRAPH_DIR = inputs/molecules-bond-sum

TEST_DATA_LST = $(INP_GRAPH_DIR)/testing_data_files_prefixes.txt
TRAIN_DATA_LST = $(INP_GRAPH_DIR)/training_data_files_prefixes.txt
VALIDATION_DATA_LST = $(INP_GRAPH_DIR)/validation_data_files_prefixes.txt

MODEL_DIR = outputs/molecules-bond-sum
RAW_TRAIN_DATA_DIR = training_data

CONTAINER = ./container.sif

VENV ?= venv

train: $(MODEL_DIR)

$(MODEL_DIR):
	$(CONTAINER) python3 run_training.py

divide: $(TEST_DATA_LST) $(TRAIN_DATA_LST) $(VALIDATION_DATA_LST)

$(TEST_DATA_LST) $(TRAIN_DATA_LST) $(VALIDATION_DATA_LST):
	find ${INP_GRAPH_DIR}/data -name '*_edges.csv' | cut -d / -f 3- | cut -d _ -f 1 | bin/divide

%.sif: %.def
	apptainer build --force $@ $<

EPOCH ?= 25
EXAMPLE ?= $(shell head -n1 $(VALIDATION_DATA_LST))

example: $(CONTAINER) $(MODEL_DIR)
	$(CONTAINER) python3 run_inference_for_one_graph.py ${MODEL_DIR}/epoch${EPOCH}.pth ${INP_GRAPH_DIR}/${EXAMPLE}_vertices_in.csv ${INP_GRAPH_DIR}/${EXAMPLE}_edges.csv /dev/stdout \
		| paste <(tail -n +2 ${INP_GRAPH_DIR}/${EXAMPLE}_vertices_out.csv) -

cleanAll distclean:
	find ${INP_GRAPH_DIR}/data -mindepth 1 -type d | xargs rm -rf
	rm -f ${TEST_DATA_LST} ${TRAIN_DATA_LST} ${VALIDATION_DATA_LST}
	rm -rf ${MODEL_DIR} ${RAW_TRAIN_DATA_DIR}
	rm -rf __pycache__
