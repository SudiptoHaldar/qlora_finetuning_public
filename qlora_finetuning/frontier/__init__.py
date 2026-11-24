"""Path B: Amazon Nova on Bedrock. Filled by req_004.

v4.01: ``config`` (settings), ``provider`` (the ``FrontierProvider`` interface), ``bedrock``
(its Converse implementation; the only module that loads boto3), ``predictor`` (the
evaluation backend), ``pricing`` (prices and inference cost), and ``baseline_report``.
Nothing is imported here, so loading the package stays light.
"""
