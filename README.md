## Content Humanizer Agent

Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
streamlit run content_humanizer_app.py
```
Optional environment variables:
`GROQ_MODEL`
`GROQ_TEMPERATURE`
`GROQ_MAX_TOKENS`
What it does
Normalizes pasted text
Analyzes sentence and vocabulary patterns
Builds a rewrite brief
Rewrites with Groq
Runs a quality gate
Performs one polishing pass if needed