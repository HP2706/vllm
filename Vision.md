# Overview

we want to create a tool that spawns a new sequence with the content of the parent sequence. Note that we opt for using tool calls instead of special tokens, as it allows more flexibility.

## PASTA-LANG Syntax
Inspired by [PASTA-LANG](https://arxiv.org/pdf/2502.11517v2) we introudce the following tags

- **`<promise/>`** - Standalone tag with attributes:
  - `topic` (string): specifies the content topic  
  - `tokens` (integer): estimated token count in multiples of 10
- **`<sync/>`** - Standalone tag that synchronizes all async threads
- For a child we ensure the following:
  the child puts the text in wants to return to the parent inside a `<return>...</return>` tag

## Our Syntax

We define 3 functions with python syntax:

```python
def spawn_child(prompt : str) -> int:
    '''
    Spawns a child note that this is async and the child will first return
    some point in the future.
    Returns the id of the child. So the model can use this id to call sync with.
    '''
```

```python
def sync(child_ids : list[str] | None = None) -> list[str]:
    '''
    Waits for child_ids or all children to return.
    '''
```

```python
def return_to_parent(text : str) -> str:
    '''
    Text to return to the parent.
    '''
```


