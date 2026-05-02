import socket
import sys

def resolve_load_balancer():
    hostname = "blatam-alb-1003572062.us-east-1.elb.amazonaws.com"
    
    try:
        result = socket.getaddrinfo(hostname, 80, socket.AF_INET)
        ips = [info[4][0] for info in result]
        unique_ips = list(set(ips))
        
        print(f"Load Balancer IP addresses for {hostname}:")
        for ip in unique_ips:
            print(f"  {ip}")
            
        print("\nHostGator DNS Configuration:")
        print("Type: A")
        print("Name: @ (or blatam.org)")
        for i, ip in enumerate(unique_ips):
            print(f"Value {i+1}: {ip}")
        print("TTL: 300 (5 minutes)")
        
        return unique_ips
        
    except Exception as e:
        print(f"Error resolving DNS: {e}")
        return []

if __name__ == "__main__":
    resolve_load_balancer()
