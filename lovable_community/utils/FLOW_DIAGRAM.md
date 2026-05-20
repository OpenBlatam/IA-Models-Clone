# Record Storage - Flow Diagrams

## Operation Flow Diagrams

### Read Operation Flow

```
┌─────────────────┐
│  read() called  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ File exists?        │
└───┬───────────┬─────┘
    │ No        │ Yes
    ▼           ▼
┌─────────┐  ┌──────────────────┐
│ Return  │  │ Open file with   │
│ []      │  │ context manager  │
└─────────┘  └────────┬──────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Load JSON     │
              └───┬───────────┘
                  │
                  ▼
          ┌───────────────┐
          │ Validate      │
          │ format        │
          └───┬───────────┘
              │
              ▼
      ┌───────────────┐
      │ Return records│
      └───────────────┘
```

### Write Operation Flow

```
┌─────────────────┐
│ write() called  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Validate input:     │
│ - Is list?          │
│ - All dicts?         │
└───┬───────────┬─────┘
    │ Invalid   │ Valid
    ▼           ▼
┌─────────┐  ┌──────────────────┐
│ Raise   │  │ Open file with   │
│ ValueError│ │ context manager  │
└─────────┘  └────────┬──────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Write JSON    │
              └───┬───────────┘
                  │
                  ▼
          ┌───────────────┐
          │ Return True   │
          └───────────────┘
```

### Update Operation Flow

```
┌─────────────────┐
│ update() called │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Validate inputs:    │
│ - record_id valid?  │
│ - updates is dict?  │
└───┬───────────┬─────┘
    │ Invalid   │ Valid
    ▼           ▼
┌─────────┐  ┌──────────────────┐
│ Raise   │  │ Read all records │
│ ValueError│ └────────┬──────────┘
└─────────┘          │
                      ▼
              ┌──────────────────┐
              │ Find record by ID│
              └───┬──────────┬───┘
                  │ Found   │ Not Found
                  ▼         ▼
          ┌───────────┐  ┌──────────┐
          │ Merge     │  │ Return   │
          │ updates   │  │ False    │
          └─────┬─────┘  └──────────┘
                │
                ▼
        ┌───────────────┐
        │ Preserve ID   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Write records │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Return True   │
        └───────────────┘
```

## Error Handling Flow

```
┌─────────────────┐
│ Operation       │
│ (read/write/    │
│  update)        │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Try block           │
└───┬─────────────────┘
    │
    ├─── Success ────► Return result
    │
    ├─── ValueError ──► Raise ValueError
    │                    (Invalid input)
    │
    ├─── JSONDecodeError ──► Raise RuntimeError
    │                       (Invalid JSON)
    │
    ├─── IOError/OSError ──► Raise RuntimeError
    │                        (File error)
    │
    └─── Other Exception ──► Raise RuntimeError
                              (Unexpected error)
```

## Context Manager Flow

```
┌─────────────────┐
│ with open(...)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ File opened         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Perform operation   │
│ (read/write)        │
└────────┬────────────┘
         │
         ├─── Success ────► File auto-closed
         │
         └─── Exception ──► File auto-closed
                            (even on error)
```

## Update Merging Flow

```
Before Update:
┌─────────────────────┐
│ Record:              │
│ {                    │
│   "id": "1",         │
│   "name": "Alice",   │
│   "age": 30          │
│ }                    │
└─────────────────────┘

Update Call:
update("1", {"age": 31})

┌─────────────────────┐
│ records[i].update(  │
│   {"age": 31}       │
│ )                    │
└─────────────────────┘

After Update:
┌─────────────────────┐
│ Record:              │
│ {                    │
│   "id": "1",         │
│   "name": "Alice",   │ ← Preserved!
│   "age": 31          │ ← Updated!
│ }                    │
└─────────────────────┘
```

## Complete CRUD Flow

```
CREATE (add)
    ┌──────────────┐
    │ add(record)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Validate     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Check exists │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Append & save│
    └──────────────┘

READ (read/get)
    ┌──────────────┐
    │ read()/get() │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Load file    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Return data  │
    └──────────────┘

UPDATE (update)
    ┌──────────────┐
    │ update(id,   │
    │  changes)    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Find record  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Merge changes│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Save file    │
    └──────────────┘

DELETE (delete)
    ┌──────────────┐
    │ delete(id)   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Filter out   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Save file    │
    └──────────────┘
```

## Data Flow

```
User Input
    │
    ▼
┌──────────────┐
│ Validation   │
└───┬──────────┘
    │
    ├─── Invalid ────► ValueError
    │
    └─── Valid ────►
                    │
                    ▼
            ┌──────────────┐
            │ File I/O     │
            │ (with context│
            │  manager)    │
            └───┬──────────┘
                │
                ├─── Success ────► Return result
                │
                └─── Error ────► RuntimeError
```

## State Transitions

```
File State Machine:

[NON-EXISTENT]
    │
    │ __init__()
    ▼
[INITIALIZED] ────► {records: []}
    │
    │ write()
    ▼
[POPULATED] ────► {records: [...]}
    │
    │ update()
    ▼
[MODIFIED] ────► {records: [updated]}
    │
    │ write()
    ▼
[SAVED] ────► Changes persisted
```


