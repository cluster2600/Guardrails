---
title:
  page: "Tracing Troubleshooting"
  nav: "Troubleshooting"
description: "Common issues and solutions for tracing in NeMo Guardrails including missing traces, connection errors, and imports."
topics: ["Guardrails"]
tags: ["Troubleshooting", "Tracing", "Debugging"]
content:
  type: "Troubleshooting"
  difficulty: "Intermediate"
  audience: ["Developer"]
---

# Troubleshooting

| Issue | Solution |
|-------|----------|
| No traces appear | Configure OpenTelemetry SDK in application code; verify `tracing.enabled: true` |
| Connection errors | Check collector is running; test with `ConsoleSpanExporter` first |
| Import errors | Install dependencies: `pip install nemoguardrails[tracing]` |
| Wrong service name | Set `Resource` with `service.name` in application code |
