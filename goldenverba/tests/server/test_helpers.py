import json

from goldenverba.server.helpers import BatchManager
from goldenverba.server.types import Credentials, DataBatchPayload


def test_batch_manager_reassembles_out_of_order_chunks():
    batch_manager = BatchManager()

    file_config_dict = {
        "fileID": "file-1",
        "filename": "example.txt",
        "isURL": False,
        "overwrite": False,
        "extension": ".txt",
        "source": "local",
        "content": "hello",
        "labels": [],
        "rag_config": {
            "Reader": {"selected": "noop", "components": {}},
            "Chunker": {"selected": "noop", "components": {}},
            "Embedder": {"selected": "noop", "components": {}},
            "Retriever": {"selected": "noop", "components": {}},
            "Generator": {"selected": "noop", "components": {}},
        },
        "file_size": 0,
        "status": "READY",
        "metadata": "",
        "status_report": {},
    }
    payload_json = json.dumps(file_config_dict, separators=(",", ":"))

    parts = [payload_json[:20], payload_json[20:60], payload_json[60:]]
    assert len(parts) == 3

    credentials = Credentials(deployment="Custom", url="localhost", key="")

    # Send chunks out of order to ensure ordering is reconstructed by index, not arrival time.
    for order in (2, 0, 1):
        payload = DataBatchPayload(
            chunk=parts[order],
            isLastChunk=False,
            total=3,
            fileID="file-1",
            order=order,
            credentials=credentials,
        )
        result = batch_manager.add_batch(payload)

    assert result is not None
    assert result.fileID == "file-1"
    assert result.filename == "example.txt"
