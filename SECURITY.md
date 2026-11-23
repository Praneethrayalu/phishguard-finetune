# Security notes

Do not upload private email bodies to a shared instance without permission. This tool does not send bodies to an external inference provider, but server operators can still access request memory. Logging middleware, proxies and analytics added by a deployer may record bodies.

Inference does not execute attachments, render HTML, follow links or retrieve URLs. Treat email text as untrusted data. There is no sandboxed attachment scanner, reputation service, authentication or enterprise boundary in this demo.

Load only artifacts you generated locally. joblib and PyTorch pickle are unsafe for untrusted files. Do not accept user-supplied model paths in an internet-facing endpoint. Dependencies should be updated and vulnerability-scanned before real deployment. This project is an educational experiment with no security service-level guarantee.

Keep tokens in environment variables or a secret store, never in commits. No API key is needed for this project's public data and model downloads.
