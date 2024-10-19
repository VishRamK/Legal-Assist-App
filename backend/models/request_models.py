from marshmallow import Schema, fields, validate

# Define your request schemas using Marshmallow

class DocumentRequestSchema(Schema):
    document = fields.Raw(required=True)  # Raw bytes of the uploaded document

class StrategyRequestSchema(Schema):
    evidence = fields.Str(required=True)  # Evidence string
    responses = fields.List(fields.Str(), required=True)  # List of previous responses

class QuestionRequestSchema(Schema):
    responses = fields.List(fields.Str(), required=True)  # List of previous responses

class JudgeRequestSchema(Schema):
    evidence = fields.Str(required=True)  # Evidence string
    responses = fields.List(fields.Str(), required=True)  # List of previous responses

# You can also define response schemas if needed for consistency
class StrategyResponseSchema(Schema):
    strategies = fields.List(fields.Str(), required=True)  # List of suggested strategies

class JudgeResponseSchema(Schema):
    probability = fields.Float(required=True)  # Probability of winning the case
