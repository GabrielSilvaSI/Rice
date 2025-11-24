import subprocess
import os
import time
import webbrowser

def run():
    """
    Inicia o backend (FastAPI) e o frontend (Streamlit) em processos separados,
    e abre o navegador na página da aplicação.
    """
    # Obter o caminho absoluto para o diretório raiz do projeto
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Caminhos para as pastas do backend e frontend
    backend_dir = os.path.join(project_root, 'backend')
    frontend_dir = os.path.join(project_root, 'frontend')
    
    # Comandos para iniciar os servidores
    # Usamos sys.executable para garantir que estamos usando o mesmo interpretador Python
    # para o streamlit, evitando problemas de ambiente.
    backend_command = ['uvicorn', 'main:app', '--reload']
    frontend_command = [os.path.join(os.path.dirname(os.sys.executable), 'streamlit.exe'), 'run', 'app.py']

    
    backend_process = None
    frontend_process = None
    
    try:
        # 1. Iniciar o servidor do backend
        print(f"Iniciando o servidor do backend em: {backend_dir}")
        backend_process = subprocess.Popen(backend_command, cwd=backend_dir)
        print("Servidor do backend iniciado. Aguardando 5 segundos para inicialização...")
        
        # Dar um tempo para o backend iniciar completamente
        time.sleep(5)
        
        # 2. Iniciar a aplicação do frontend
        print(f"Iniciando a aplicação do frontend em: {frontend_dir}")
        frontend_process = subprocess.Popen(frontend_command, cwd=frontend_dir)
        print("Aplicação do frontend iniciada.")
        
        # 3. Abrir o navegador
        print("Abrindo o navegador em http://localhost:8501...")
        webbrowser.open("http://localhost:8501")
        
        print("\n✅ Aplicação rodando. Pressione Ctrl+C neste terminal para parar ambos os servidores.")
        
        # Manter o script principal rodando para poder encerrar os filhos
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nSinal de encerramento recebido. Parando os servidores...")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        # 4. Encerrar os processos filhos ao sair
        if frontend_process:
            frontend_process.terminate()
            print("Aplicação do frontend parada.")
        if backend_process:
            backend_process.terminate()
            print("Servidor do backend parado.")
        print("Limpeza concluída.")

if __name__ == "__main__":
    run()