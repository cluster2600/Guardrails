---
title:
  page: "About Configuring Guardrails"
  nav: "About Configuring Guardrails"
description: "Configure YAML files, Colang flows, custom actions, and other components to control LLM behavior."
topics: ["Configuration", "AI Safety"]
tags: ["Configuration", "YAML", "Colang", "Actions", "Setup"]
content:
  type: "Overview"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# About Configuring Guardrails

This section explains how to configure your guardrails system, from defining LLM models and guardrail flows in YAML to implementing advanced features like Colang flows and custom actions.

---

## Before You Begin with Configuring Guardrails

Before diving into configuring guardrails, ensure you have the required components ready and understand the overall structure of the guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Prerequisites
:link: before-configuration
:link-type: doc

Prepare LLM endpoints, NemoGuard NIMs, and knowledge base documents before configuration.
:::

:::{grid-item-card} Configuration Overview
:link: overview
:link-type: doc

Learn the structure of guardrails configuration files and how components work together.
:::

::::

---

## Core Configuration

Configure the essential components of your guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Configuring YAML File
:link: yaml-schema/index
:link-type: doc

Define models, guardrails, prompts, and tracing settings in the config.yml file.
:::

:::{grid-item-card} YAML Schema Reference
:link: configuration-reference
:link-type: doc

Complete reference for all config.yml options including models, rails, prompts, and advanced settings.
:::

:::{grid-item-card} Guardrails Catalog
:link: guardrail-catalog
:link-type: doc

Browse the complete catalog of built-in guardrails including content safety, jailbreak detection, and third-party integrations.
:::

:::{grid-item-card} Colang Flows
:link: colang/index
:link-type: doc

Learn Colang, the event-driven language for defining guardrails flows and bot behavior.
:::

::::

---

## Advanced Configuration

Optional configurations for extending and optimizing your guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Custom Actions
:link: actions/index
:link-type: doc

Create and register custom Python actions callable from Colang flows.
:::

:::{grid-item-card} Custom Initialization
:link: custom-initialization/index
:link-type: doc

Use config.py to register custom LLM providers, embedding providers, and shared resources at startup.
:::

:::{grid-item-card} Knowledge Base & Providers
:link: other-configurations/index
:link-type: doc

Configure knowledge base folders for RAG and custom embedding search providers.
:::

:::{grid-item-card} Caching
:link: caching/index
:link-type: doc

Configure in-memory caching and KV cache reuse to improve performance and reduce latency.
:::

:::{grid-item-card} Exceptions and Error Handling
:link: exceptions
:link-type: doc

Raise and handle exceptions in guardrails flows to control error behavior and custom responses.
:::

::::
