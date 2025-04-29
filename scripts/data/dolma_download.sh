DATA_DIR="/home/nlp/achimoa/workspace/hebrew_text_retrieval/data/dolma/v1_7"
PARALLEL_DOWNLOADS="24"
DOLMA_VERSION="v1_7"

git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"


# cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"
cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget --show-progress --progress=bar:force:noscroll -x -ncH --cut-dirs=1 -P "$DATA_DIR"

