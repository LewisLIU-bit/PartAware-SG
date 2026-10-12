# Workspace driver archive

These are the exact inspected Windows-workspace adapters used for this research.
They require the saved layout `workspace/v20_work`, `workspace/scan200_work`,
`workspace/scan200_expansion_work`, and `workspace/esam_hypersim_work`.
Copy the v20 driver files back into that layout before execution. Their ROOT
is the workspace; running them directly inside this archive is unsupported.
Ordinary algorithm modules live one level up in tools/research and take Context
inputs. Models/datasets/credentials are external and are never included here.
This archive does not change the unique production registry or default v19.
