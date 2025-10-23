# Configuration Changes Log

This document tracks changes to configuration files and schemas to maintain reproducibility across versions.

## Format

```
## [Date] - Config Name
### Changed
- Description of change
- Impact on experiments

### Migration
- How to migrate old configs
```

## [2025-10-23] - Initial Schema with Pydantic Validation

### Added
- Pydantic schemas for config validation (`src/knapsack_gnn/config/schemas.py`)
- ExperimentConfig with nested schemas:
  - ModelConfig: Model architecture settings
  - TrainingConfig: Hyperparameters and optimization settings
  - DataConfig: Dataset generation parameters
  - LoggingConfig: Logging and checkpointing settings
  - ReproducibilityConfig: Determinism flags

### Changed
- All YAMLs must now conform to Pydantic schemas
- Extra fields will raise validation errors (stricter validation)
- Seed validation: must be in range [0, 2^32 - 1]

### Migration
Existing configs should mostly work unchanged, but:
1. Run `make validate-configs` to check for issues
2. Unknown fields will be rejected (remove or add to schema)
3. Invalid types/ranges will be caught early

---

## Guidelines for Future Changes

When modifying configuration structure:

1. **Document the change** in this file with:
   - Date and description
   - Fields added/removed/modified
   - Default values
   - Impact on reproducibility

2. **Update schema** in `src/knapsack_gnn/config/schemas.py`

3. **Provide migration path** for existing configs:
   - Script to auto-migrate if possible
   - Manual steps if needed

4. **Version compatibility**:
   - Note which experiments need re-running
   - Mark configs as deprecated if removing fields

5. **Examples**:
   - Update `configs/train_default.yaml` to reflect changes
   - Add comments explaining new fields

---

## Template for New Entries

```markdown
## [YYYY-MM-DD] - Brief Description

### Added
- New field `foo.bar` (default: value) - Purpose

### Changed
- Field `baz` renamed to `qux`
- Field `old_field` type changed from int to float

### Removed
- Deprecated field `obsolete_setting`

### Migration
- Old configs will work but should update:
  - Rename `baz` to `qux`
  - Remove `obsolete_setting`
- Run: `python scripts/migrate_config.py old_config.yaml`

### Impact
- Experiments using old configs can be reproduced
- New experiments should use updated schema
```
