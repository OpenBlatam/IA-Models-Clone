import socket
import json
import time
import threading
import uuid
import sys
from concurrent.futures import ThreadPoolExecutor

HOST = "127.0.0.1"
PORT = 50051
TOTAL_REQUESTS = 5000
CONCURRENCY = 100

def send_syscall(pid, priority, call_type, payload):
    try:
        req = {
            "pid": pid,
            "priority": priority,
            "call_type": call_type,
            "payload": payload
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(json.dumps(req).encode('utf-8'))
            data = s.recv(4096)
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        return {"status": "error", "message": str(e)}

def worker(worker_id, num_requests):
    success_count = 0
    error_count = 0
    pid = f"stress_worker_{worker_id}_{uuid.uuid4().hex[:4]}"
    
    for i in range(num_requests):
        # We will stress test the DashMap by doing MEM_WRITE and MEM_READ rapidly
        key = f"key_{worker_id}_{i}"
        
        # 1. WRITE
        payload_write = json.dumps({"key": key, "value": f"stress_data_{i}"})
        res_w = send_syscall(pid, 1, "SYS_MEM_WRITE", payload_write)
        if res_w.get("status") == "success":
            success_count += 1
        else:
            error_count += 1

        # 2. READ
        res_r = send_syscall(pid, 1, "SYS_MEM_READ", key)
        if res_r.get("status") == "success":
            success_count += 1
        else:
            error_count += 1
            
    return success_count, error_count

def run_stress_test():
    print(f"--- Iniciando Stress Test del Kernel (Rust) ---")
    print(f"-> Objetivos: DashMap L1 Cache & Tokio Async Router")
    print(f"-> Concurrencia: {CONCURRENCY} Hilos")
    print(f"-> Peticiones Totales (Reads + Writes): {TOTAL_REQUESTS * 2}")
    
    start_time = time.time()
    
    total_success = 0
    total_errors = 0
    requests_per_worker = TOTAL_REQUESTS // CONCURRENCY
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(worker, i, requests_per_worker) for i in range(CONCURRENCY)]
        
        for future in futures:
            success, errors = future.result()
            total_success += success
            total_errors += errors

    end_time = time.time()
    duration = end_time - start_time
    total_processed = total_success + total_errors
    rps = total_processed / duration if duration > 0 else 0
    
    print("\n" + "="*40)
    print("RESULTADOS DEL STRESS TEST")
    print("="*40)
    print(f"Tiempo Total   : {duration:.3f} segundos")
    print(f"Exitos         : {total_success}")
    print(f"Errores        : {total_errors}")
    print(f"Rendimiento    : {rps:.2f} SysCalls por segundo (RPS)")
    print("="*40)

if __name__ == "__main__":
    run_stress_test()
