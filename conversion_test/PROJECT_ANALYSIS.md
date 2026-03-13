# Conversion Test Project — Full Analysis

---

## 1. Experimental Setting

### 1.1 Purpose

The project evaluates large language models (LLMs) on **unit and conversion tasks** under several conditions: with or without an in-prompt conversion guide, and (where applicable) on a **math-only** formulation of the same conversion. The goal is to compare **reasoning** vs **non-reasoning** models and to separate performance on "understanding the task" from "doing the arithmetic." We find so far that models perform well overall on math-only tasks, but falter when the same question is asked in a natural language (or more casual) format. Adding a conversion guide helps, although this is not an intuitive or natural step that a user would take.

### 1.2 Pipeline Overview

- **Preprocessing** (`preprocessing.py`): Reads JSON conversion configs (e.g. `conversions/volume.json`, `conversions/timezone.json`) and `conversions/numbers.json`, generates prompts and gold answers, and writes TSV files under `full_results/preprocessed/`.
- **Inference** (`main_conversion.py`): For each (model × domain × condition), loads the preprocessed TSV, calls the model via LiteLLM, parses `<answer>...</answer>` tags (or raw text), computes loss (exact or tolerance-based), and writes results under `full_results/<condition_dir>/<model>/`.

### 1.3 Conditions

With-guide and no-guide use **the same set of prompts** (same conversions, same inputs); the only difference is that the with-guide prompt includes the conversion guide and the no-guide prompt omits it.

| Condition                | Suffix       | Output directory     | Description |
|--------------------------|--------------|----------------------|-------------|
| **in_domain_with_guide** | (none)       | `results/`           | Full conversion prompt **with** an in-prompt conversion guide (factors, tables, or formulas). |
| **in_domain_no_guide**   | `_no_guide`  | `results_no_guide/`  | Same task, **without** the conversion guide in the prompt. |
| **math_only**            | `_math_only` | `results_math_only/` | Same conversion expressed as a single math expression (e.g. `what is 1*1000` for liters→ml, `what is (1+3)%24` for timezone). Only supported for linear, currency, temperature, and timezone — **not** for clothing (table lookup). |

### 1.4 Domains

| Category | Domains | Conversion type | Math-only? |
|----------|---------|-----------------|------------|
| **Numeric/linear** | volume, speed, density, energy, bits_bytes, moles_to_particles | Linear factor multiplication | Yes |
| **Cooking** | cooking | Linear factor (with substance distractors, e.g. "1 cup of flour") | Yes |
| **Currency** | currency | Exchange rate multiplication | Yes |
| **Temperature** | temperature | Formula-based (e.g. F = C×9/5 + 32) | Yes |
| **Timezone** | timezone | City-to-city time offset (e.g. LA → NY) | Yes |
| **Clothing** | men's/women's clothing, shoe, pant, bra sizes | Discrete lookup table (US ↔ EUR, etc.) | **No** |

### 1.5 Models

| Type | Models | Notes |
|------|--------|-------|
| **Non-reasoning** | `gpt-4o`, `claude-haiku-4-5`, `qwen-coder`, `llama-4`, `gpt-oss-120b` | temperature=0, no chain-of-thought |
| **Reasoning** | `gpt-5.2`, `deepseek-v3.1`, `qwen3-235b-thinking`, `qwen3-next-thinking` | Reasoning/thinking enabled; responses include internal reasoning then final answer |

---

## 2. Exact Prompts Used for Each Type of Model

### 2.1 System Prompts

**Non-reasoning, non-timezone:**

```
You are a precise unit conversion expert. Provide only the numerical answer
with up to 4 decimal places, within <answer> and </answer> tags
(e.g., <answer>42.5</answer>).
```

**Non-reasoning, timezone:**

```
You are a precise timezone conversion expert. Provide the time in the same
format as the input (e.g., 1AM, 3:49PM), within <answer> and </answer> tags
(e.g., <answer>1AM</answer>, <answer>3:49PM</answer>).
```

**Reasoning, non-timezone:**

```
You are a precise conversion expert. Provide your final answer within
<answer> and </answer> tags. For numerical answers, provide only the number
(e.g., <answer>42.5</answer>). For size answers, provide only the size
(e.g., <answer>M</answer> or <answer>32</answer>).
```

**Reasoning, timezone:**

```
You are a precise timezone conversion expert. Provide your final answer
within <answer> and </answer> tags in the same format as the input
(e.g., <answer>1AM</answer>, <answer>3:49PM</answer>).
```

### 2.2 User Prompts by Condition

- **With guide:** Task sentence + conversion guide (factors/table/formulas) + output instruction.
- **No guide:** Same task sentence, no guide, same output instruction.
- **Math only:** Single line, e.g. `what is 1*1000`, `what is (5*9/5)+32`, `what is (1+3.0)%24`.

---

## 3. Per-Domain Examples: Non-Reasoning (gpt-4o) vs Reasoning (qwen3-235b-thinking)

For each domain below, we show **the same input** given to both a non-reasoning model (**gpt-4o**) and a reasoning model (**qwen3-235b-thinking**), with full raw responses. Where applicable, the **math-only equivalent** prompt and response from the **same model** is shown alongside.

---

### 3.1 Volume (linear) — 1 liter → milliliters, gold: 1000

**With-guide prompt** (identical for both models):

```
Convert 1 liters to milliliters.

Conversion guide:
Conversion factors (relative to liters):
  cups: 4.22675
  gallons: 0.264172
  liters: 1.0
  milliliters: 1000.0
  ounces: 33.814023
  pints: 2.11338
  quarts: 1.05669
  tablespoons: 67.628045
  teaspoons: 202.884135

Provide only the numerical value.
```

**Math-only equivalent prompt:** `what is 1*1000`

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `<answer>1000.0</answer>` | Yes |
| **Math only** | `<answer>1000</answer>` | Yes |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, let's see. I need to convert 1 liter to milliliters. The user provided a conversion
guide with factors relative to liters. Let me check that.

Looking at the conversion factors, it says milliliters: 1000.0. Oh, right, because 1 liter
is equal to 1000 milliliters. So if I have 1 liter, multiplying by 1000.0 should give me
the milliliters. Let me confirm. Yeah, the standard conversion is 1 L = 1000 mL. The guide
confirms that the factor for milliliters relative to liters is 1000.0. So 1 times 1000.0
is 1000. The answer should be 1000.
[/REASONING]
<answer>1000.0</answer>
```

**Math only — full trace:**

```
[REASONING]
Okay, the user is asking what 1 multiplied by 1000 is. That seems straightforward. Let me
think... 1 times any number is just that number itself, right? So 1 * 1000 should be 1000.

Hmm, the user also included some specific instructions about how to format the answer.
They want the final answer inside <answer> tags, and for numerical answers, just the number
without any extra text. ...

The calculation is definitely 1000. No decimals, no units, just a whole number. So
according to their rules, I should output <answer>1000</answer>.
[/REASONING]
<answer>1000</answer>
```

---

### 3.2 Temperature (formula) — 5 °C → °F, gold: 41

**With-guide prompt** (identical for both models):

```
Convert 5 degrees Celsius to degrees Fahrenheit.

Conversion guide:
Temperature conversion formulas:
  Celsius to Fahrenheit: F = (C × 9/5) + 32
  Fahrenheit to Celsius: C = (F - 32) × 5/9
  Celsius to Kelvin: K = C + 273.15
  Kelvin to Celsius: C = K - 273.15
  Fahrenheit to Kelvin: K = (F - 32) × 5/9 + 273.15
  Kelvin to Fahrenheit: F = (K - 273.15) × 9/5 + 32

Provide only the numerical value.
```

**Math-only equivalent prompt:** `what is (5*9/5)+32`

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `41.0` | Yes |
| **Math only** | `<answer>41</answer>` | Yes |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, so I need to convert 5 degrees Celsius to Fahrenheit. Let me check the conversion
formula provided. The guide says Celsius to Fahrenheit is F = (C × 9/5) + 32. Alright,
so C is 5 here.

First, multiply 5 by 9/5. Let me do that step by step. 5 times 9 is 45, and then divide
by 5. So 45 divided by 5 is 9. Then add 32 to that result. So 9 plus 32 equals 41. Wait,
is that right? Let me double-check. 5°C times 9/5 is indeed 9, then plus 32 makes 41°F.
Hmm, I think that's correct. Let me verify with another method. I know that 0°C is 32°F
and 100°C is 212°F. So each degree Celsius is 1.8 degrees Fahrenheit. So 5°C should be
5*1.8 = 9, plus 32 is 41. Yeah, that seems right. I think the answer is 41.
[/REASONING]
<answer>41</answer>
```

**Math only — full trace:**

```
[REASONING]
Okay, let's see. The problem is (5*9/5)+32. ... First, I remember the order of operations:
parentheses, exponents, multiplication and division (from left to right), then addition
and subtraction. So here, the parentheses are around 5*9/5, so I should do that part first.

Starting with 5 multiplied by 9. 5*9 is 45. Then, divide that by 5. So 45 divided by 5.
Let me calculate that. 45/5 is 9. ... So the part inside the parentheses simplifies to 9.

Now, add 32 to that result. So 9 + 32. Let's do that. 9 plus 32 is 41. ... Yeah, that
seems correct. I think that's the answer.
[/REASONING]
<answer>41</answer>
```

---

### 3.3 Timezone (city offset) — 1AM Los Angeles → New York, gold: 4AM

**With-guide prompt** (identical for both models):

```
Convert 1AM in Los Angeles time to New York time. Assume you are thinking
about standard time, not daylight savings.

Conversion guide:
City timezones (relative to GMT):
  Hilo: GMT-10.0
  Los Angeles: GMT-8.0
  Salem: GMT-8.0
  Yorba Linda: GMT-8.0
  New York: GMT-5.0
  Altoona: GMT-5.0
  London: GMT+0.0
  Llieda: GMT+1.0
  Kafue: GMT+2.0
  Istanbul: GMT+3.0
  Delhi: GMT+5.5
  Noorabad: GMT+5.5
  Dhaka: GMT+6.0
  Jakarta: GMT+7.0
  Shanghai: GMT+8.0
  Manila: GMT+8.0
  Zhemuzhen: GMT+8.0
  Tokyo: GMT+9.0
  Nagai: GMT+9.0
  Lavrentiya: GMT+12.0

Provide the time in the same format (e.g., 1AM, 3:49PM).
```

**Math-only equivalent prompt:** `what is (1+3.0)%24`
*(1AM = hour 1; NY offset − LA offset = −5 − (−8) = +3; result 4 → 4AM)*

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `4AM` | Yes |
| **Math only** | `<answer>4.0</answer>` | Yes |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, so I need to convert 1AM Los Angeles time to New York time, and they
specified to use standard time, not daylight savings. Let me check the time
zones given.

First, looking at the conversion guide. Los Angeles is GMT-8.0, and New York
is GMT-5.0. So the difference between them should be the difference in their
offsets from GMT.

New York is GMT-5, Los Angeles is GMT-8. So to find the time difference,
subtract LA's offset from NY's. Wait, actually, since both are behind GMT, the
time difference is the difference between their offsets. So NY is 3 hours ahead
of LA because 8 - 5 = 3. Wait, no: if LA is GMT-8, that means when it's 12PM
GMT, it's 4AM in LA. And NY is GMT-5, so 12PM GMT is 7AM in NY. So NY is 3
hours ahead of LA.

Wait, let me think again. If it's 1AM in LA (GMT-8), then GMT time would be
1AM + 8 hours = 9AM GMT. Then New York is GMT-5, so 9AM GMT minus 5 hours is
4AM. ...

Wait, but another way: the time difference between LA and NY is 3 hours (since
8 - 5 = 3). Since NY is east of LA, it's ahead. So 1AM + 3 hours = 4AM. Yes,
that's correct.

So the answer should be 4AM.
[/REASONING]
<answer>4AM</answer>
```

**Math only — full trace (excerpt):**

```
[REASONING]
Okay, let's see. The user is asking what (1 + 3.0) % 24 is. ... First, let's
compute 1 + 3.0. 1 plus 3.0 is 4.0 ... Now, modulo operation with 24. When
you take a number modulo 24, it's like finding the remainder when divided by
24. But since 4.0 is less than 24, the remainder should just be 4.0 itself.
... So 4.0 - 24 * floor(4.0 / 24) = 4.0 - 24*0 = 4.0. So the result is 4.0.
[/REASONING]
<answer>4.0</answer>
```

---

### 3.4 Currency (exchange rate) — 10 USD → EUR, gold: 8.5

**With-guide prompt** (identical for both models):

```
Convert 10 USD to EUR.

Conversion guide:
Exchange rates (relative to USD):
  GBP: 0.73
  CHF: 0.78
  EUR: 0.85
  USD: 1.0
  CAD: 1.37
  AUD: 1.44
  CNH: 6.94
  SLL: 24.48
  JPY: 155.5
  GNF: 8751.0
  UZS: 12250.0
  IDR: 16766.0
  LAK: 21495.0
  VND: 26019.92
  LBP: 85550.0
  IRR: 1189712.0

Provide only the numerical value.
```

**Math-only equivalent prompt:** `what is 10*0.85`

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `8.5` | Yes |
| **Math only** | `8.5` | Yes |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, let's see. I need to convert 10 USD to EUR. The conversion guide says the
exchange rate for EUR relative to USD is 0.85. Hmm, so that means 1 USD is equal
to 0.85 EUR.

Wait, right, because the rates are given as how much of the foreign currency you
get for 1 USD. So if EUR is 0.85, then each dollar converts to 0.85 euros.

So for 10 USD, I just multiply 10 by 0.85. Let me check that. 10 * 0.85 equals
8.5. So 10 USD should be 8.5 EUR.

Let me make sure I didn't mix up the direction. Sometimes exchange rates can be
quoted as how much USD you get for 1 EUR, but the problem states "Exchange rates
(relative to USD)", so it's EUR per USD. So yes, 1 USD = 0.85 EUR. Therefore,
multiplying 10 by 0.85 gives the correct amount in EUR.

No other steps needed here. The answer should be 8.5.
[/REASONING]
<answer>8.5</answer>
```

**Math only — full trace:**

```
[REASONING]
Okay, the user is asking for the result of 10 multiplied by 0.85. Let me think
about this.

First, I know that multiplying by 10 is straightforward. When you multiply a
decimal by 10, you move the decimal point one place to the right. So 0.85 times
10 would be 8.5.

Let me double-check that. 0.85 is the same as 85/100. If I multiply 85/100 by
10, that's (85 * 10) / 100 = 850 / 100 = 8.5. Yep, that checks out. ...

The result is 8.5, so I'll present that as the answer.
[/REASONING]
<answer>8.5</answer>
```

---

### 3.5 Cooking (linear, with substance distractor) — 1 cup of flour → gallons, gold: 0.0625

This domain uses the same linear factors as volume but **includes substance distractors** (e.g. "of flour") that should not affect the numeric conversion but may confuse models.

**With-guide prompt** (identical for both models):

```
Convert 1 cup of flour to gallon.

Conversion guide:
Conversion factors (relative to cups):
  cups: 1.0
  fluid_ounces: 8.0
  gallons: 0.0625
  liters: 0.236588
  milliliters: 236.588
  pints: 0.5
  quarts: 0.25
  tablespoons: 16.0
  teaspoons: 48.0

Provide only the numerical value.
```

**Math-only equivalent prompt:** `what is 1*0.0625`
*(Note: the math-only prompt strips the distractor "flour" — it is pure arithmetic.)*

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `0.0625` | Yes |
| **Math only** | `<answer>0.0625</answer>` | Yes |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, let's see. I need to convert 1 cup of flour to gallons. The user provided
a conversion guide with factors relative to cups. Let me check that.

Looking at the guide, it says gallons: 0.0625. Wait, so the conversion factor
from cups to gallons is 0.0625. That means 1 cup equals 0.0625 gallons. Let me
confirm. Since there are 16 cups in a gallon (because 1 gallon is 4 quarts,
1 quart is 2 pints, 1 pint is 2 cups, so 4*2*2=16 cups per gallon). So 1 cup
is 1/16 of a gallon, which is 0.0625. Yep, that matches the given factor. So
1 cup is 0.0625 gallons. The answer should be 0.0625.
[/REASONING]
<answer>0.0625</answer>
```

**Math only — full trace (excerpt):**

```
[REASONING]
Okay, so I need to figure out what 1 multiplied by 0.0625 is. ... First,
multiplying any number by 1 should just give the number itself, right? So
1 times anything is that anything. So maybe 1 * 0.0625 is just 0.0625. ...

Let me verify: 0.0625 is 1/16. Because 1 divided by 16 is 0.0625. 16 × 0.0625
= 1.0. Yes, so 0.0625 is indeed 1/16. So if I multiply 1 by 1/16, that's just
1/16, which is 0.0625. So that checks out.
[/REASONING]
<answer>0.0625</answer>
```

---

### 3.6 Clothing (table lookup) — Men's shoe US 6 → EUR, gold: 38

There is **no math-only condition** for clothing — the conversion is a discrete table lookup, not a formula.

**With-guide prompt** (identical for both models):

```
Convert men's shoe size 6 in US to EUR sizing.

Conversion guide:
Size conversion table (US to EUR):
  6 → 38
  6.5 → 38.5
  7 → 39
  7.5 → 40
  8 → 41
  8.5 → 42
  9 → 43
  9.5 → 43.25
  10 → 43.5
  10.5 → 44
  11 → 44.5
  11.5 → 45
  12 → 46
  12.5 → 46.5
  13 → 47
  14 → 48.5

Provide only the size and nothing else.
```

#### gpt-4o (non-reasoning)

| Condition | Raw response | Correct? |
|-----------|-------------|----------|
| **With guide** | `38.0` | Yes |
| **No guide** | `<answer>38.5</answer>` | **No** (gold is 38) |
| **Math only** | N/A | — |

#### qwen3-235b-thinking (reasoning)

**With guide — full trace:**

```
[REASONING]
Okay, so I need to convert a men's US shoe size 6 to EUR sizing. Let me check
the conversion guide provided.

Looking at the table, it lists US sizes and their corresponding EUR sizes. Let me
scan through the list:

- US 6 → EUR 38
- US 6.5 → 38.5
- US 7 → 39
- And so on.

The user is asking for US size 6. According to the first entry in the table, US 6
converts directly to EUR 38. There's no mention of any exceptions or additional
notes here, so it should be straightforward.

I should make sure there's no confusion between men's and women's sizing, but the
question specifically states it's men's shoe size, and the conversion table
provided seems to be for men's since the sizes start at US 6 which is typical for
men (women's sizes are usually higher in US but lower in EUR). ...

Double-checking the table again: yes, the first line is 6 → 38. So the answer
should be 38.
[/REASONING]
<answer>38</answer>
```

---

## 4. Summary: Same Input, Both Models, All Conditions

| Domain | Input | Gold | gpt-4o (with guide) | gpt-4o (math only) | qwen3-235b (with guide) | qwen3-235b (math only) |
|--------|-------|------|---------------------|---------------------|--------------------------|------------------------|
| Volume | 1 L → ml | 1000 | 1000.0 | 1000 | 1000.0 | 1000 |
| Temperature | 5°C → °F | 41 | 41.0 | 41 | 41 | 41 |
| Timezone | 1AM LA → NY | 4AM | 4AM | 4.0 | 4AM | 4.0 |
| Currency | 10 USD → EUR | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 |
| Cooking | 1 cup flour → gal | 0.0625 | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| Clothing | US 6 → EUR | 38 | 38.0 | N/A | 38 | N/A |

---

## 5. References in Code

- **System prompts:** `config.py` → `get_system_prompt(is_timezone, is_reasoning)`.
- **User prompt construction:** `preprocessing.py` → `create_prompt(...)` (with_guide / no_guide / math_only), and `create_conversion_guide(...)` for the guide text.
- **API call:** `api.py` → `call_model(model_name, prompt, domain)` builds messages with the appropriate system prompt and the user prompt from the TSV.
- **Conditions and paths:** `config.py` → `CONDITIONS`, `DEFAULT_BASE_DIR`, `PREPROCESSED_SUBDIR`.
