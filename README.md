# Sprint Session – 4º Semestre 2025/1

**Tema:** Prevenção de Fraudes  
**Empresa Parceira:** Mercado Livre

---

## 📁 Estrutura do Projeto

- `data/`: dados brutos.   
- `requirements.txt`: dependências do projeto.

---

## 🚀 Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone <https://github.com/insper-classroom/2025-1-app-mercado-de-fraude-app>
   cd 2025-1-app-mercado-de-fraude-app
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure as credenciais da AWS para acessar o bucket S3:**

   **Linux/Mac:**
   ```bash
   export AWS_ACCESS_KEY_ID="SUA_KEY"
   export AWS_SECRET_ACCESS_KEY="SUA_SECRET"
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:AWS_ACCESS_KEY_ID="SUA_KEY"
   $env:AWS_SECRET_ACCESS_KEY="SUA_SECRET"
   ```

4. **Rodar a aplicação com Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## 👥 Equipe

- Ana Beatriz da Cunha (MLEng)  
- Isabela Rodrigues (MLOps)  
- Manoela Saragoça (MLEng)  
- Gustavo Lagoa (MLOps)
