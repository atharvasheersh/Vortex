# Resident-bounded amendment to the ten-million protocol

Defined on 11 November 2026 after preserving the first unrestricted mapped run
as resource_limited. That run built and opened all ten million vectors, completed
all 60 exact measurements and the one-probe condition, then stopped during the
four-probe condition when available physical memory fell below 384 MiB. Its
peak working set reached approximately 3.14 GiB. It is not a completed sweep.

Repeat the same full protocol in fresh directories, including a fresh 1m control,
with an additional **1536 MiB maximum resident working set** for each benchmark
process. Apply Windows SetProcessWorkingSetSizeEx with HARDWS_MAX_ENABLE and
HARDWS_MIN_DISABLE, verify the returned limit/flags, and restore original settings
on exit. Affect only the benchmark process. Do not change other applications,
machine power settings, the original runner, the free-memory guard or corpus size.

This controls resident process pages, including mapped pages. It does not cap
committed virtual memory or the total OS file cache. Demand paging and working-set
replacement are inside query wall time. Standby pages may remain in RAM; this is
not proof of cold SSD access or a physical-RAM-exceeding corpus. Capture the policy
and observed working-set peaks. Both final sizes must use the same policy and
executed sources. Retain the unrestricted control and partial 10m run separately;
do not combine their timings with the bounded pair.

All vector/query seeds, dimensions, list counts, training choices, independent
references, repetitions, warmups, probe gate and measurement checks in
[the original protocol](capacity_10m_spec.md) remain the same. No primary bounded
10m timings have been collected when defining this amendment.

Windows API semantics: [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex).
