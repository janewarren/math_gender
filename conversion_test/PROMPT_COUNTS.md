# Prompt Counts Per Domain

This document explains how many prompts are generated for each domain in the full experiment.

## Formula

For **regular** and **no-guide** modes:
```
Prompts per domain = (unit_pairs) × (test_values) × (1 + num_contexts)
```

Where:
- `1` = context-free prompt (distractor = null)
- `num_contexts` = number of contexts/distractors (0 if none)

For **math-only** mode:
```
Prompts per domain = (unit_pairs) × (test_values) × 1
```
(No contexts are used in math-only mode)

## Test Values

- **Most domains**: 200 values (100 easy + 100 hard from `numbers.json`)
- **Currency**: 200 values (100 easy + 100 hard) for ALL unit pairs
- **Timezone**: Uses times from `times.json` (typically 100 easy + 100 hard = 200 times)
- **Clothing sizes**: Uses values from `size_mappings` (varies by size type, typically 10-20 values)

## Contexts (Distractors)

- **Density, Volume, Moles_to_particles**: 32 contexts from `substances.json`
- **Energy, Cooking**: Contexts defined in their JSON files (varies)
- **Clothing sizes**: No contexts (0)
- **Other domains**: No contexts (0)

## Domain-by-Domain Breakdown

### 1. bits_bytes
- **Unit pairs**: 6
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 0
- **Regular/No-guide**: `6 × 200 × 1 = 1,200`
- **Math-only**: `6 × 200 × 1 = 1,200`

### 2. clothing_sizes_men_shoe_size
- **Unit pairs**: Check `clothing_sizes_men.json`
- **Test values**: ~10-20 (from size_mappings)
- **Contexts**: 0
- **Regular/No-guide**: `unit_pairs × ~15 × 1 = unit_pairs × ~15`
- **Math-only**: Not supported (clothing sizes don't support math-only)

### 3. clothing_sizes_men_pant_size
- **Unit pairs**: Check `clothing_sizes_men.json`
- **Test values**: ~10-20 (from size_mappings)
- **Contexts**: 0
- **Regular/No-guide**: `unit_pairs × ~15 × 1 = unit_pairs × ~15`
- **Math-only**: Not supported

### 4. clothing_sizes_women_shoe_size
- **Unit pairs**: Check `clothing_sizes_women.json`
- **Test values**: ~10-20 (from size_mappings)
- **Contexts**: 0
- **Regular/No-guide**: `unit_pairs × ~15 × 1 = unit_pairs × ~15`
- **Math-only**: Not supported

### 5. clothing_sizes_women_bra_size
- **Unit pairs**: Check `clothing_sizes_women.json`
- **Test values**: ~10-20 (from size_mappings)
- **Contexts**: 0
- **Regular/No-guide**: `unit_pairs × ~15 × 1 = unit_pairs × ~15`
- **Math-only**: Not supported

### 6. clothing_sizes_women_pant_size
- **Unit pairs**: Check `clothing_sizes_women.json`
- **Test values**: ~10-20 (from size_mappings)
- **Contexts**: 0
- **Regular/No-guide**: `unit_pairs × ~15 × 1 = unit_pairs × ~15`
- **Math-only**: Not supported

### 7. cooking
- **Unit pairs**: 6
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 30 (ingredients from `cooking.json`)
- **Regular/No-guide**: `6 × 200 × (1 + 30) = 6 × 200 × 31 = 37,200`
- **Math-only**: `6 × 200 × 1 = 1,200`

### 8. currency
- **Unit pairs**: 9 (from `currency.json`)
- **Test values**: 200 (100 easy + 100 hard) for ALL pairs
- **Contexts**: 0
- **Regular/No-guide**: `9 × 200 × 1 = 1,800`
- **Math-only**: `9 × 200 × 1 = 1,800`

### 9. density
- **Unit pairs**: 6 (from `density.json`)
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 32 (from `substances.json`)
- **Regular/No-guide**: `6 × 200 × (1 + 32) = 6 × 200 × 33 = 39,600`
- **Math-only**: `6 × 200 × 1 = 1,200`

### 10. energy
- **Unit pairs**: 10 (from `energy.json`)
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 30 (from `energy.json`)
- **Regular/No-guide**: `10 × 200 × (1 + 30) = 10 × 200 × 31 = 62,000`
- **Math-only**: `10 × 200 × 1 = 2,000`

### 11. moles_to_particles
- **Unit pairs**: 2
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 32 (from `substances.json`)
- **Regular/No-guide**: `2 × 200 × (1 + 32) = 2 × 200 × 33 = 13,200`
- **Math-only**: `2 × 200 × 1 = 400`

### 12. speed
- **Unit pairs**: 6
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 0
- **Regular/No-guide**: `6 × 200 × 1 = 1,200`
- **Math-only**: `6 × 200 × 1 = 1,200`

### 13. temperature
- **Unit pairs**: 6 (from `temperature.json`)
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 0
- **Regular/No-guide**: `6 × 200 × 1 = 1,200`
- **Math-only**: `6 × 200 × 1 = 1,200`

### 14. timezone
- **Unit pairs**: 10 (from `timezone.json`)
- **Test values**: ~200 (from `times.json`, typically 100 easy + 100 hard times)
- **Contexts**: 0
- **Regular/No-guide**: `10 × 200 × 1 = 2,000`
- **Math-only**: `10 × 200 × 1 = 2,000`

### 15. volume
- **Unit pairs**: 6 (from `volume.json`)
- **Test values**: 200 (100 easy + 100 hard)
- **Contexts**: 32 (from `substances.json`)
- **Regular/No-guide**: `6 × 200 × (1 + 32) = 6 × 200 × 33 = 39,600`
- **Math-only**: `6 × 200 × 1 = 1,200`

## Summary Table

| Domain | Unit Pairs | Test Values | Contexts | Regular/No-guide | Math-only |
|--------|------------|-------------|----------|-------------------|-----------|
| bits_bytes | 6 | 200 | 0 | **1,200** | **1,200** |
| clothing_sizes_men_shoe | ~5-10 | ~15 | 0 | ~75-150 | N/A |
| clothing_sizes_men_pant | ~5-10 | ~15 | 0 | ~75-150 | N/A |
| clothing_sizes_women_shoe | ~5-10 | ~15 | 0 | ~75-150 | N/A |
| clothing_sizes_women_bra | ~5-10 | ~15 | 0 | ~75-150 | N/A |
| clothing_sizes_women_pant | ~5-10 | ~15 | 0 | ~75-150 | N/A |
| cooking | 6 | 200 | 30 | **37,200** | **1,200** |
| currency | 9 | 200 | 0 | **1,800** | **1,800** |
| density | 6 | 200 | 32 | **39,600** | **1,200** |
| energy | 10 | 200 | 30 | **62,000** | **2,000** |
| moles_to_particles | 2 | 200 | 32 | **13,200** | **400** |
| speed | 6 | 200 | 0 | **1,200** | **1,200** |
| temperature | 6 | 200 | 0 | **1,200** | **1,200** |
| timezone | 10 | 200 | 0 | **2,000** | **2,000** |
| volume | 6 | 200 | 32 | **39,600** | **1,200** |

## Notes

1. **Clothing sizes** have variable test values depending on available size mappings
2. **Math-only mode** is not supported for clothing sizes
3. **Contexts multiply** the prompt count significantly for domains like density, volume, energy, and cooking
4. **Currency** uses all 200 numbers for all 9 pairs (not per pair)
5. To get exact counts, check the actual JSON files for unit_pairs and contexts

## Total Prompts (Approximate)

**Regular/No-guide total**: ~200,000 - 250,000 prompts
- Largest contributors: energy (62,000), density (39,600), volume (39,600), cooking (37,200)
- Clothing sizes: ~375-750 total (varies by available size mappings)

**Math-only total**: ~15,000 - 20,000 prompts
- Much smaller because contexts are excluded
- Clothing sizes not included (math-only not supported)
