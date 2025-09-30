# Диаграммы архитектуры NormaML

## Общая архитектура системы

```mermaid
graph TB
    subgraph "User Interface Layer"
        API[Public API]
        CLI[Command Line Interface]
        CFG[Configuration Manager]
    end
    
    subgraph "Core Engine"
        PM[Plugin Manager]
        DM[Data Manager<br/>Zero-Copy]
        EE[Execution Engine<br/>Multi-Mode]
        ET[Experiment Tracker<br/>MLflow]
        EB[Event Bus]
    end
    
    subgraph "Plugin Layer"
        DL[Data Loaders]
        FE[Feature Engineers]
        MT[Model Trainers]
        EV[Evaluators]
        BE[Backends]
    end
    
    subgraph "Execution Environments"
        AS[AsyncIO Engine]
        PR[Process Pool]
        DC[Docker Containers]
    end
    
    subgraph "Data Storage"
        SHM[Shared Memory]
        MMP[Memory Maps]
        ARW[Arrow Buffers]
        MLF[MLflow Storage]
    end
    
    API --> PM
    CLI --> CFG
    CFG --> PM
    
    PM --> DL
    PM --> FE
    PM --> MT
    PM --> EV
    PM --> BE
    
    PM --> EE
    DM --> SHM
    DM --> MMP
    DM --> ARW
    
    EE --> AS
    EE --> PR
    EE --> DC
    
    ET --> MLF
    
    EB --> PM
    EB --> DM
    EB --> EE
    EB --> ET
    
    DL --> DM
    FE --> DM
    MT --> DM
    EV --> DM
    
    AS --> ET
    PR --> ET
    DC --> ET
```

## Plugin System Architecture

```mermaid
graph LR
    subgraph "Plugin Discovery"
        EP[Entry Points]
        DEC[Decorators]
        CFG[Config-based]
    end
    
    subgraph "Plugin Registry"
        REG[Central Registry]
        META[Metadata Store]
        DEP[Dependency Graph]
        CAP[Capability Index]
    end
    
    subgraph "Plugin Lifecycle"
        INIT[Initialization]
        EXEC[Execution]
        CLEAN[Cleanup]
    end
    
    subgraph "Plugin Types"
        DL[Data Loaders]
        FE[Feature Engineers]
        MT[Model Trainers]
        EV[Evaluators]
        BE[ML Backends]
    end
    
    EP --> REG
    DEC --> REG
    CFG --> REG
    
    REG --> META
    REG --> DEP
    REG --> CAP
    
    META --> INIT
    DEP --> INIT
    CAP --> INIT
    
    INIT --> EXEC
    EXEC --> CLEAN
    
    EXEC --> DL
    EXEC --> FE
    EXEC --> MT
    EXEC --> EV
    EXEC --> BE
```

## Zero-Copy Data Flow

```mermaid
graph TD
    subgraph "Data Sources"
        CSV[CSV Files]
        PAR[Parquet Files]
        DB[Databases]
        API_SRC[APIs]
    end
    
    subgraph "Data Loading"
        LAZY[Polars LazyFrame]
        STREAM[Streaming Loader]
    end
    
    subgraph "Memory Management"
        SHM[Shared Memory<br/>POSIX SHM]
        MMP[Memory Mapped<br/>Files]
        ARW[Arrow Buffers<br/>In-Memory]
        COW[Copy-on-Write<br/>Views]
    end
    
    subgraph "Processing"
        P1[Plugin Process 1]
        P2[Plugin Process 2]
        P3[Plugin Process 3]
        DC1[Docker Container 1]
        DC2[Docker Container 2]
    end
    
    subgraph "Zero-Copy Operations"
        VIEW[Data Views]
        REF[References]
        PTR[Memory Pointers]
    end
    
    CSV --> LAZY
    PAR --> LAZY
    DB --> STREAM
    API_SRC --> STREAM
    
    LAZY --> SHM
    STREAM --> MMP
    
    SHM --> VIEW
    MMP --> REF
    ARW --> PTR
    
    VIEW --> P1
    REF --> P2
    PTR --> P3
    
    SHM --> DC1
    MMP --> DC2
    
    P1 --> COW
    P2 --> COW
    P3 --> COW
    
    DC1 --> COW
    DC2 --> COW
```

## Parallel Execution Architecture

```mermaid
graph TB
    subgraph "Task Scheduler"
        TQ[Task Queue<br/>Priority-based]
        DEP[Dependency<br/>Resolver]
        LB[Load Balancer]
        RM[Resource Manager]
    end
    
    subgraph "Execution Engines"
        AE[AsyncIO Engine<br/>I/O Bound]
        PE[Process Engine<br/>CPU Bound]
        DE[Docker Engine<br/>Isolated]
    end
    
    subgraph "Resource Pools"
        TP[Thread Pool]
        PP[Process Pool]
        DP[Docker Pool]
    end
    
    subgraph "Shared Resources"
        SHM[Shared Memory]
        IPC[IPC Channels]
        VOL[Docker Volumes]
    end
    
    subgraph "Monitoring"
        STAT[Statistics]
        LOG[Logging]
        METRIC[Metrics]
    end
    
    TQ --> DEP
    DEP --> LB
    LB --> RM
    
    RM --> AE
    RM --> PE
    RM --> DE
    
    AE --> TP
    PE --> PP
    DE --> DP
    
    TP --> SHM
    PP --> SHM
    DP --> VOL
    
    PP --> IPC
    DP --> IPC
    
    AE --> STAT
    PE --> LOG
    DE --> METRIC
```

## MLflow Integration Flow

```mermaid
graph LR
    subgraph "Experiment Setup"
        EXP[Create Experiment]
        RUN[Start Run]
        NEST[Nested Runs]
    end
    
    subgraph "Auto Logging"
        HOOK[Method Hooks]
        PARAM[Parameters]
        METRIC[Metrics]
        TAG[Tags]
    end
    
    subgraph "Manual Logging"
        MODEL[Models]
        ARTIFACT[Artifacts]
        DATA[Data Info]
        RESOURCE[Resources]
    end
    
    subgraph "Distributed Tracking"
        COORD[Coordinator]
        CHILD[Child Runs]
        AGG[Aggregation]
    end
    
    subgraph "Storage"
        MLF_DB[MLflow Database]
        ART_STORE[Artifact Store]
        MODEL_REG[Model Registry]
    end
    
    EXP --> RUN
    RUN --> NEST
    
    RUN --> HOOK
    HOOK --> PARAM
    HOOK --> METRIC
    HOOK --> TAG
    
    RUN --> MODEL
    RUN --> ARTIFACT
    RUN --> DATA
    RUN --> RESOURCE
    
    RUN --> COORD
    COORD --> CHILD
    CHILD --> AGG
    
    PARAM --> MLF_DB
    METRIC --> MLF_DB
    TAG --> MLF_DB
    
    MODEL --> MODEL_REG
    ARTIFACT --> ART_STORE
    
    AGG --> MLF_DB
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant PluginManager
    participant DataManager
    participant ExecutionEngine
    participant MLflow
    participant Plugin
    
    User->>API: Submit ML Task
    API->>PluginManager: Discover Required Plugins
    PluginManager->>DataManager: Register Data Sources
    DataManager->>DataManager: Create Shared Memory Views
    
    API->>MLflow: Start Experiment Run
    MLflow->>MLflow: Setup Auto-logging
    
    API->>ExecutionEngine: Schedule Task
    ExecutionEngine->>PluginManager: Initialize Plugin
    PluginManager->>Plugin: Create Instance
    
    ExecutionEngine->>DataManager: Get Data Handle
    DataManager->>ExecutionEngine: Return Zero-Copy Handle
    
    ExecutionEngine->>Plugin: Execute Method
    Plugin->>MLflow: Log Parameters (auto)
    Plugin->>DataManager: Process Data (zero-copy)
    Plugin->>MLflow: Log Metrics (auto)
    Plugin->>ExecutionEngine: Return Results
    
    ExecutionEngine->>DataManager: Store Results
    ExecutionEngine->>MLflow: Log Artifacts
    ExecutionEngine->>API: Task Complete
    
    API->>MLflow: End Run
    API->>User: Return Results
```

## Data Pipeline Flow

```mermaid
flowchart TD
    START([Start Pipeline]) --> LOAD[Load Data]
    
    LOAD --> CHECK_SIZE{Data Size Check}
    CHECK_SIZE -->|Small < 100MB| MEMORY[Load to Memory]
    CHECK_SIZE -->|Medium 100MB-1GB| SHARED[Load to Shared Memory]
    CHECK_SIZE -->|Large > 1GB| MMAP[Memory Map File]
    
    MEMORY --> PROCESS[Process Data]
    SHARED --> PROCESS
    MMAP --> STREAM[Stream Processing]
    
    STREAM --> CHUNK[Process Chunks]
    CHUNK --> COMBINE[Combine Results]
    COMBINE --> PROCESS
    
    PROCESS --> FEATURE[Feature Engineering]
    FEATURE --> SELECT[Feature Selection]
    SELECT --> TRANSFORM[Data Transform]
    
    TRANSFORM --> SPLIT[Train/Test Split]
    SPLIT --> TRAIN[Model Training]
    TRAIN --> VALIDATE[Model Validation]
    
    VALIDATE --> EVALUATE[Model Evaluation]
    EVALUATE --> SAVE[Save Model]
    SAVE --> END([End Pipeline])
    
    subgraph "Zero-Copy Operations"
        PROCESS
        FEATURE
        SELECT
        TRANSFORM
        SPLIT
    end
    
    subgraph "MLflow Tracking"
        TRAIN
        VALIDATE
        EVALUATE
        SAVE
    end
```

## Security and Isolation Model

```mermaid
graph TB
    subgraph "Security Layers"
        AUTH[Authentication]
        AUTHZ[Authorization]
        AUDIT[Audit Logging]
    end
    
    subgraph "Process Isolation"
        PROC1[Process 1<br/>Plugin A]
        PROC2[Process 2<br/>Plugin B]
        PROC3[Process 3<br/>Plugin C]
    end
    
    subgraph "Container Isolation"
        CONT1[Container 1<br/>Isolated Environment]
        CONT2[Container 2<br/>Resource Limited]
        CONT3[Container 3<br/>Network Isolated]
    end
    
    subgraph "Data Access Control"
        SHM_ACL[Shared Memory ACL]
        FILE_PERM[File Permissions]
        NET_POLICY[Network Policies]
    end
    
    subgraph "Resource Limits"
        CPU_LIMIT[CPU Limits]
        MEM_LIMIT[Memory Limits]
        DISK_LIMIT[Disk Limits]
        TIME_LIMIT[Time Limits]
    end
    
    AUTH --> PROC1
    AUTH --> CONT1
    AUTHZ --> SHM_ACL
    AUDIT --> FILE_PERM
    
    PROC1 --> SHM_ACL
    PROC2 --> SHM_ACL
    PROC3 --> SHM_ACL
    
    CONT1 --> NET_POLICY
    CONT2 --> CPU_LIMIT
    CONT3 --> MEM_LIMIT
    
    CPU_LIMIT --> PROC1
    MEM_LIMIT --> PROC2
    DISK_LIMIT --> CONT1
    TIME_LIMIT --> CONT2
```

## Performance Optimization Flow

```mermaid
graph LR
    subgraph "Data Optimization"
        TYPE_OPT[Type Optimization<br/>Int64->Int8]
        COMPRESS[Compression<br/>LZ4/ZSTD]
        COLUMNAR[Columnar Format<br/>Arrow/Parquet]
    end
    
    subgraph "Memory Optimization"
        LAZY[Lazy Loading]
        CACHE[Smart Caching]
        GC[Garbage Collection]
    end
    
    subgraph "Execution Optimization"
        PARALLEL[Parallelization]
        VECTORIZE[Vectorization]
        BATCH[Batching]
    end
    
    subgraph "I/O Optimization"
        ASYNC[Async I/O]
        PREFETCH[Prefetching]
        BUFFER[Buffering]
    end
    
    TYPE_OPT --> LAZY
    COMPRESS --> CACHE
    COLUMNAR --> GC
    
    LAZY --> PARALLEL
    CACHE --> VECTORIZE
    GC --> BATCH
    
    PARALLEL --> ASYNC
    VECTORIZE --> PREFETCH
    BATCH --> BUFFER
```

## Error Handling and Recovery

```mermaid
graph TD
    ERROR[Error Detected] --> CLASSIFY{Error Classification}
    
    CLASSIFY -->|Transient| RETRY[Retry Logic]
    CLASSIFY -->|Resource| SCALE[Scale Resources]
    CLASSIFY -->|Data| FALLBACK[Fallback Strategy]
    CLASSIFY -->|Plugin| REPLACE[Replace Plugin]
    CLASSIFY -->|Fatal| FAIL[Fail Gracefully]
    
    RETRY --> SUCCESS{Success?}
    SUCCESS -->|Yes| CONTINUE[Continue Processing]
    SUCCESS -->|No| ESCALATE[Escalate Error]
    
    SCALE --> RESOURCE_CHECK{Resources Available?}
    RESOURCE_CHECK -->|Yes| CONTINUE
    RESOURCE_CHECK -->|No| QUEUE[Queue for Later]
    
    FALLBACK --> ALT_STRATEGY[Alternative Strategy]
    ALT_STRATEGY --> CONTINUE
    
    REPLACE --> PLUGIN_SEARCH[Find Alternative Plugin]
    PLUGIN_SEARCH --> PLUGIN_FOUND{Plugin Found?}
    PLUGIN_FOUND -->|Yes| CONTINUE
    PLUGIN_FOUND -->|No| FAIL
    
    ESCALATE --> LOG[Log Error]
    QUEUE --> LOG
    FAIL --> LOG
    
    LOG --> NOTIFY[Notify User]
    NOTIFY --> CLEANUP[Cleanup Resources]
    CLEANUP --> END([End])
    
    CONTINUE --> END
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_IDE[IDE/Editor]
        DEV_TEST[Local Testing]
        DEV_DEBUG[Debugging]
    end
    
    subgraph "CI/CD Pipeline"
        BUILD[Build & Test]
        PACKAGE[Package Creation]
        DEPLOY[Deployment]
    end
    
    subgraph "Production Environment"
        LOAD_BAL[Load Balancer]
        API_GATEWAY[API Gateway]
        WORKERS[Worker Nodes]
    end
    
    subgraph "Storage Layer"
        MLFLOW_DB[MLflow Database]
        ARTIFACT_STORE[Artifact Storage]
        DATA_LAKE[Data Lake]
    end
    
    subgraph "Monitoring"
        METRICS[Metrics Collection]
        LOGGING[Centralized Logging]
        ALERTS[Alerting]
    end
    
    DEV_IDE --> BUILD
    DEV_TEST --> BUILD
    BUILD --> PACKAGE
    PACKAGE --> DEPLOY
    
    DEPLOY --> LOAD_BAL
    LOAD_BAL --> API_GATEWAY
    API_GATEWAY --> WORKERS
    
    WORKERS --> MLFLOW_DB
    WORKERS --> ARTIFACT_STORE
    WORKERS --> DATA_LAKE
    
    WORKERS --> METRICS
    WORKERS --> LOGGING
    METRICS --> ALERTS
    LOGGING --> ALERTS
```

## Configuration Management

```mermaid
graph LR
    subgraph "Configuration Sources"
        ENV[Environment Variables]
        FILE[Config Files<br/>YAML/JSON]
        CLI[Command Line Args]
        DEFAULT[Default Values]
    end
    
    subgraph "Configuration Manager"
        PARSER[Config Parser]
        VALIDATOR[Validator]
        MERGER[Config Merger]
        RESOLVER[Variable Resolver]
    end
    
    subgraph "Configuration Types"
        CORE[Core Config]
        PLUGIN[Plugin Config]
        RUNTIME[Runtime Config]
        SECRET[Secret Config]
    end
    
    subgraph "Configuration Consumers"
        PM[Plugin Manager]
        DM[Data Manager]
        EE[Execution Engine]
        ET[Experiment Tracker]
    end
    
    ENV --> PARSER
    FILE --> PARSER
    CLI --> PARSER
    DEFAULT --> PARSER
    
    PARSER --> VALIDATOR
    VALIDATOR --> MERGER
    MERGER --> RESOLVER
    
    RESOLVER --> CORE
    RESOLVER --> PLUGIN
    RESOLVER --> RUNTIME
    RESOLVER --> SECRET
    
    CORE --> PM
    PLUGIN --> PM
    RUNTIME --> DM
    SECRET --> EE
    
    CORE --> ET
    PLUGIN --> ET
```

Эти диаграммы показывают:

1. **Общую архитектуру** - взаимодействие основных компонентов
2. **Plugin систему** - обнаружение, регистрация и выполнение
3. **Zero-copy data flow** - эффективная передача данных
4. **Параллельное выполнение** - управление ресурсами и задачами
5. **MLflow интеграцию** - отслеживание экспериментов
6. **Взаимодействие компонентов** - последовательность операций
7. **Data pipeline** - обработка данных от начала до конца
8. **Безопасность** - изоляция и контроль доступа
9. **Оптимизацию производительности** - стратегии ускорения
10. **Обработку ошибок** - стратегии восстановления
11. **Развертывание** - производственная среда
12. **Управление конфигурацией** - источники и потребители настроек

Архитектура обеспечивает модульность, масштабируемость и производительность при сохранении простоты использования и надежности.