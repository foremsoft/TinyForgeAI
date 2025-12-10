# GitHub Discussions Setup Guide

This guide explains how to enable and configure GitHub Discussions for community support.

## Enabling Discussions

1. Go to your repository **Settings**
2. Scroll down to **Features** section
3. Check **Discussions** to enable it
4. Click **Set up discussions** to create initial categories

## Recommended Categories

Configure these discussion categories for a well-organized community:

| Category | Description | Format |
|----------|-------------|--------|
| **Announcements** | Project updates and releases | Announcement |
| **Q&A** | Technical questions and answers | Question/Answer |
| **Ideas** | Feature suggestions and feedback | Open-ended |
| **Show and Tell** | Share your projects using TinyForgeAI | Open-ended |
| **General** | General discussions about the project | Open-ended |

### Creating Categories

1. Go to **Discussions** tab
2. Click the gear icon (⚙️) → **Edit categories**
3. Add each category with appropriate format:
   - **Announcements**: Only maintainers can post
   - **Q&A**: Enable "Mark answers" feature
   - **Ideas**: Good for feature requests
   - **Show and Tell**: Community showcase
   - **General**: Catch-all discussions

## Category Descriptions

Copy these descriptions when creating categories:

**Announcements**
```
Official project announcements, releases, and important updates from the maintainers.
```

**Q&A**
```
Get help with TinyForgeAI. Ask questions about installation, training, deployment, or troubleshooting.
```

**Ideas**
```
Share your ideas for new features, improvements, or integrations. Vote on ideas you'd like to see implemented.
```

**Show and Tell**
```
Share what you've built with TinyForgeAI! Show off your models, integrations, or creative applications.
```

**General**
```
General discussions about TinyForgeAI, tiny language models, and related topics.
```

## Discussion Templates

Create issue/discussion templates in `.github/DISCUSSION_TEMPLATE/`:

### Q&A Template (`q-a.yml`)

```yaml
title: "[Q&A] "
labels: ["question"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for asking a question! Please provide details to help us assist you.

  - type: textarea
    id: question
    attributes:
      label: Your Question
      description: What would you like help with?
    validations:
      required: true

  - type: textarea
    id: context
    attributes:
      label: Context
      description: Any additional context (code, error messages, etc.)

  - type: dropdown
    id: area
    attributes:
      label: Area
      options:
        - Installation
        - Training
        - Inference
        - Deployment
        - Documentation
        - Other
```

### Ideas Template (`ideas.yml`)

```yaml
title: "[Idea] "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Share your feature idea or improvement suggestion!

  - type: textarea
    id: idea
    attributes:
      label: Your Idea
      description: Describe the feature or improvement
    validations:
      required: true

  - type: textarea
    id: use-case
    attributes:
      label: Use Case
      description: How would this feature be used?

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Any alternative solutions you've considered?
```

## Pinned Discussions

Pin these discussions for visibility:

1. **Welcome & Getting Started** - Introduction to the community
2. **FAQ** - Frequently asked questions
3. **Contributing Guidelines** - How to contribute

## Community Guidelines

Add community guidelines as a pinned discussion:

```markdown
# Community Guidelines

Welcome to the TinyForgeAI community!

## Be Respectful
- Treat everyone with respect and kindness
- No harassment, discrimination, or offensive content
- Assume good intentions

## Be Helpful
- Share knowledge and help others learn
- Provide constructive feedback
- Search before posting duplicate questions

## Stay On Topic
- Keep discussions relevant to TinyForgeAI
- Use appropriate categories for your posts
- Avoid spam and self-promotion

## Report Issues
- Use Issues for bugs, not Discussions
- Report code of conduct violations to maintainers

Thank you for being part of our community!
```

## Moderation

### Enabling Moderation Features

1. Go to **Settings** → **Moderation options**
2. Enable:
   - **Limit interactions** for new users if needed
   - **Code of conduct** (use Contributor Covenant)

### Converting Discussions

- **Discussion → Issue**: For bugs reported in Q&A
- **Issue → Discussion**: For questions filed as issues

## Integration with README

Add Discussions link to README:

```markdown
## Community

- [GitHub Discussions](https://github.com/foremsoft/TinyForgeAI/discussions) - Questions and community support
- [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues) - Bug reports and feature requests
```

## Analytics

Monitor community health through:
- **Insights** → **Community** for discussion metrics
- Track question resolution rates
- Identify common issues for documentation improvements
