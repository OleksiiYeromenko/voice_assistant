"""Process-wide state shared across modules.

The app is started with ``python -m src.main``, which runs that file as ``__main__``,
while other modules (the state machine) do ``from src.main import ...`` and so load a
second copy of it as ``src.main``. A module-level variable set in one copy is invisible
to the other, so state that both sides need lives here: this module is imported by its
real name from both and is only ever loaded once.
"""

import threading

# Set while the assistant is idle (waiting for a trigger). A local session summary waits
# for it so it does not compete with the user's query on the Pi's single llama.cpp slot.
assistant_idle = threading.Event()
assistant_idle.set()

# The _LocalCachePrimer created at startup (None until then / when there is no local
# llama.cpp backend). Used to re-prime the local prompt cache after a local summary.
local_primer = None
