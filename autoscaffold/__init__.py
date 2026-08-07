"""AutoScaffold: Teacher-written scaffold text injected into training prompts only.

See DESIGN.md (port contract) and INTEGRATION.md (upstream touch points).
Modules are import-light on purpose: nothing here pulls torch/ray/vllm, so the
orchestrator and tests run without the training environment.
"""
