import inference

class DummySummarizer:
    def __call__(self, text, max_length=150, min_length=30, do_sample=False):
        return [{"summary_text": f"Summary of: {text}"}]

class DummySentiment:
    def __call__(self, text):
        return [{"label": "POSITIVE", "score": 0.99}]

class DummyBedrockClient:
    def invoke_model(self, body, modelId, accept, contentType):
        class DummyResponse:
            def get(self, key):
                if key == "body":
                    import io, json
                    return io.BytesIO(json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode())
        return DummyResponse()

if __name__ == "__main__":
    # Mock model components
    model_components = {
        "summarizer": DummySummarizer(),
        "sentiment_analyzer": DummySentiment(),
        "bedrock_client": DummyBedrockClient()
    }

    # Example input
    input_data = [
        "This is a test document for summarization and sentiment analysis.",
        "Another example input for embedding generation."
    ]

    results = inference.predict_fn(input_data, model_components)
    for result in results:
        print(result)
