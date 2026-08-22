"""
In-memory data store for CV Quality app.
In production: replace with PostgreSQL + S3.
"""
import uuid
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

STORAGE_ROOT = Path("/tmp/cv_quality")
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


def new_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass
class Project:
    id: str
    name: str
    description: str
    created_at: float
    image_count: int = 0
    normal_count: int = 0
    defect_count: int = 0
    status: str = "active"  # active, archived


@dataclass
class TrainingJob:
    id: str
    project_id: str
    model_name: str
    algorithm: str
    backbone: str
    status: str  # queued, preparing, running, completed, failed
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: int = 0
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    id: str
    project_id: str
    name: str
    algorithm: str
    backbone: str
    version: str
    created_at: float
    status: str  # ready, evaluating, deployed
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    device: str = "CPU"


@dataclass
class InferenceResult:
    id: str
    model_id: str
    project_id: str
    image_path: str
    anomaly_score: float
    decision: str  # normal, anomalous
    threshold: float
    inference_ms: float
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    created_at: float = 0.0
    device: str = "CPU"


class CVStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._projects: Dict[str, Project] = {}
        self._jobs: Dict[str, TrainingJob] = {}
        self._models: Dict[str, Model] = {}
        self._results: List[InferenceResult] = []
        self._alerts: List[Dict] = []
        self._seed_demo_data()

    def _seed_demo_data(self):
        """Seed with example projects for demo purposes."""
        p1 = Project(
            id="demo1",
            name="Bearing Surface Inspection",
            description="Quality control for industrial bearing surfaces",
            created_at=time.time() - 86400 * 3,
            image_count=120,
            normal_count=100,
            defect_count=20,
            status="active",
        )
        p2 = Project(
            id="demo2",
            name="PCB Defect Detection",
            description="Printed circuit board anomaly detection",
            created_at=time.time() - 86400 * 7,
            image_count=85,
            normal_count=70,
            defect_count=15,
            status="active",
        )
        m1 = Model(
            id="model1",
            project_id="demo1",
            name="Bearing-PatchCore-v1",
            algorithm="PatchCore",
            backbone="resnet18",
            version="1.0",
            created_at=time.time() - 86400 * 2,
            status="deployed",
            metrics={"image_auroc": 0.94, "ap": 0.91, "f1": 0.89, "threshold": 0.62},
            threshold=0.62,
            device="CPU",
        )
        m2 = Model(
            id="model2",
            project_id="demo2",
            name="PCB-PatchCore-v1",
            algorithm="PatchCore",
            backbone="resnet18",
            version="1.0",
            created_at=time.time() - 86400 * 5,
            status="ready",
            metrics={"image_auroc": 0.91, "ap": 0.88, "f1": 0.86, "threshold": 0.58},
            threshold=0.58,
            device="CPU",
        )
        for p in [p1, p2]:
            self._projects[p.id] = p
        for m in [m1, m2]:
            self._models[m.id] = m
        # Seed monitoring data (simulated inspection history)
        import random
        rng = random.Random(42)
        now = time.time()
        for i in range(50):
            score = rng.uniform(0.1, 0.95)
            decision = "anomalous" if score > 0.62 else "normal"
            self._results.append(InferenceResult(
                id=new_id(),
                model_id="model1",
                project_id="demo1",
                image_path="",
                anomaly_score=round(score, 3),
                decision=decision,
                threshold=0.62,
                inference_ms=rng.uniform(80, 250),
                created_at=now - rng.uniform(0, 86400 * 7),
                device="CPU",
            ))

    # ── Projects ──────────────────────────────────────────────────────────
    def create_project(self, name: str, description: str = "") -> Project:
        with self._lock:
            p = Project(id=new_id(), name=name, description=description, created_at=time.time())
            self._projects[p.id] = p
            return p

    def get_project(self, pid: str) -> Optional[Project]:
        return self._projects.get(pid)

    def list_projects(self) -> List[Project]:
        return sorted(self._projects.values(), key=lambda p: p.created_at, reverse=True)

    def update_project_counts(self, pid: str, normal: int, defect: int):
        with self._lock:
            p = self._projects.get(pid)
            if p:
                p.normal_count = normal
                p.defect_count = defect
                p.image_count = normal + defect

    # ── Training Jobs ─────────────────────────────────────────────────────
    def create_job(self, project_id: str, model_name: str, algorithm: str, backbone: str) -> TrainingJob:
        with self._lock:
            job = TrainingJob(
                id=new_id(),
                project_id=project_id,
                model_name=model_name,
                algorithm=algorithm,
                backbone=backbone,
                status="queued",
                created_at=time.time(),
            )
            self._jobs[job.id] = job
            return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def append_log(self, job_id: str, msg: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── Models ────────────────────────────────────────────────────────────
    def create_model(self, project_id: str, name: str, algorithm: str,
                     backbone: str, checkpoint_path: str, metrics: dict,
                     threshold: float, device: str) -> Model:
        with self._lock:
            m = Model(
                id=new_id(),
                project_id=project_id,
                name=name,
                algorithm=algorithm,
                backbone=backbone,
                version="1.0",
                created_at=time.time(),
                status="ready",
                checkpoint_path=checkpoint_path,
                metrics=metrics,
                threshold=threshold,
                device=device,
            )
            self._models[m.id] = m
            return m

    def get_model(self, mid: str) -> Optional[Model]:
        return self._models.get(mid)

    def list_models(self, project_id: Optional[str] = None) -> List[Model]:
        models = list(self._models.values())
        if project_id:
            models = [m for m in models if m.project_id == project_id]
        return sorted(models, key=lambda m: m.created_at, reverse=True)

    # ── Inference Results ─────────────────────────────────────────────────
    def add_result(self, result: InferenceResult):
        with self._lock:
            self._results.append(result)
            # Keep only last 1000
            if len(self._results) > 1000:
                self._results = self._results[-1000:]

    def get_results(self, project_id: Optional[str] = None, limit: int = 100) -> List[InferenceResult]:
        results = self._results
        if project_id:
            results = [r for r in results if r.project_id == project_id]
        return sorted(results, key=lambda r: r.created_at, reverse=True)[:limit]

    def monitoring_stats(self, project_id: Optional[str] = None) -> dict:
        results = self.get_results(project_id, limit=500)
        if not results:
            return {"total": 0, "ok_rate": 0, "nok_rate": 0, "avg_score": 0, "trend": []}
        total = len(results)
        nok = sum(1 for r in results if r.decision == "anomalous")
        ok = total - nok
        avg_score = sum(r.anomaly_score for r in results) / total
        # Daily trend (last 7 days)
        import time as t
        now = t.time()
        trend = []
        for day in range(6, -1, -1):
            day_start = now - (day + 1) * 86400
            day_end = now - day * 86400
            day_results = [r for r in results if day_start <= r.created_at <= day_end]
            day_nok = sum(1 for r in day_results if r.decision == "anomalous")
            trend.append({
                "day": 6 - day,
                "total": len(day_results),
                "nok": day_nok,
                "ok": len(day_results) - day_nok,
            })
        return {
            "total": total,
            "ok": ok,
            "nok": nok,
            "ok_rate": round(ok / total * 100, 1),
            "nok_rate": round(nok / total * 100, 1),
            "avg_score": round(avg_score, 3),
            "avg_latency_ms": round(sum(r.inference_ms for r in results) / total, 1),
            "trend": trend,
        }

    # ── Alerts ────────────────────────────────────────────────────────────
    def get_alerts(self) -> List[Dict]:
        return self._alerts[-50:]

    def add_alert(self, level: str, message: str, project_id: Optional[str] = None):
        with self._lock:
            self._alerts.append({
                "id": new_id(),
                "level": level,
                "message": message,
                "project_id": project_id,
                "timestamp": time.time(),
                "read": False,
            })


# Singleton store instance
_store = CVStore()


def get_store() -> CVStore:
    return _store
