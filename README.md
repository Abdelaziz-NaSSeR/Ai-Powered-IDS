<<<<<<< HEAD
# Ai-Powered-IDS
The IDS integrated with AI model (Semi-IPS) is an advanced network security system that combines artificial intelligence with intrusion detection capabilities to provide real-time network threat monitoring and response.
=======
# AI_IDS_PROJECT

Professional project layout for semi-supervised intrusion detection (CICFlowMeter → preprocessing → semi-supervised detection → API).

## Quick start

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare data and run preprocessing (notebooks or scripts).

3. Train model:
   python scripts/train_save.py <preprocessed_csv> <label_col>

4. Start API:
   uvicorn api.main:app --reload

5. Test endpoints:
   GET /, GET /health, POST /api/v1/predict
>>>>>>> dbcf700 (Initial commit: Add AI-Powered IDS project files)
