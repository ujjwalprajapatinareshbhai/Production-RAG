
import { useEffect, useState } from 'react'
import './App.css'

function App() {
  // ==========================================
  // 1. LOGIN FORM STATE
  // ==========================================

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [tenantId, setTenantId] = useState('')

  // ==========================================
  // 2. AUTHENTICATION STATE
  // ==========================================

  const [user, setUser] = useState(() => {
    const token = sessionStorage.getItem('access_token')

    if (!token) return null

    return {
      access_token: token,
      tenant_id: sessionStorage.getItem('tenant_id'),
      role: sessionStorage.getItem('user_role'),
    }
  })

  // ==========================================
  // 3. LOGIN STATE
  // ==========================================

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // ==========================================
  // 4. DASHBOARD STATE
  // ==========================================

  const [documents, setDocuments] = useState([])
  const [health, setHealth] = useState('Checking...')
  const [readiness, setReadiness] = useState('Checking...')
  const [dashboardError, setDashboardError] = useState('')

  // ==========================================
  // 5. SEARCH STATE
  // ==========================================

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)

  // ==========================================
  // 6. LOGIN FUNCTION
  // ==========================================

  async function handleLogin(event) {
    event.preventDefault()

    setError('')
    setLoading(true)

    try {
      const response = await fetch('/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          password,
          tenant_id: tenantId,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : JSON.stringify(data.detail || 'Login failed')
        )
      }

      // Store authentication information
      sessionStorage.setItem('access_token', data.access_token)
      sessionStorage.setItem('tenant_id', data.tenant_id)
      sessionStorage.setItem('user_role', data.role)

      setUser({
        access_token: data.access_token,
        tenant_id: data.tenant_id,
        role: data.role,
      })

    } catch (err) {
      setError(err.message || 'Unable to connect to server.')
    } finally {
      setLoading(false)
    }
  }

  // ==========================================
  // 7. LOAD DASHBOARD DATA
  // ==========================================

  useEffect(() => {
    if (!user) return

    async function loadDashboard() {
      const headers = {
        Authorization: `Bearer ${user.access_token}`,
      }

      try {
        const results = await Promise.allSettled([
          fetch('/documents', { headers }),
          fetch('/health'),
          fetch('/ready'),
        ])

        // Documents
        const documentsResult = results[0]

        if (documentsResult.status === 'fulfilled') {
          const response = documentsResult.value

          if (response.ok) {
            const data = await response.json()
            setDocuments(data.documents || [])
          } else if (response.status === 401) {
            handleLogout()
            return
          } else {
            setDashboardError('Could not load documents.')
          }
        } else {
          setDashboardError('Could not connect to documents API.')
        }

        // API Health
        const healthResult = results[1]

        if (
          healthResult.status === 'fulfilled' &&
          healthResult.value.ok
        ) {
          const data = await healthResult.value.json()
          setHealth(data.status || 'Available')
        } else {
          setHealth('Unavailable')
        }

        // System Readiness
        const readyResult = results[2]

        if (
          readyResult.status === 'fulfilled' &&
          readyResult.value.ok
        ) {
          const data = await readyResult.value.json()
          setReadiness(data.status || 'Ready')
        } else {
          setReadiness('Not ready')
        }

      } catch {
        setDashboardError(
          'An error occurred while loading the dashboard.'
        )
      }
    }

    loadDashboard()
  }, [user])

  // ==========================================
  // 8. SEARCH DOCUMENTS FUNCTION
  // ==========================================

  async function handleSearch(event) {
    event.preventDefault()

    // Validate search input
    if (!searchQuery.trim()) {
      setSearchError('Please enter a search query.')
      return
    }

    // Prepare search state
    setSearchLoading(true)
    setSearchError('')
    setSearchResults([])
    setHasSearched(false)

    try {
      // Send authenticated search request
      const response = await fetch('/search', {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${user.access_token}`,
        },

        body: JSON.stringify({
          query: searchQuery.trim(),
          limit: 5,
          candidate_limit: 20,
        }),
      })

      const data = await response.json()

      // Check whether authentication has expired
      if (response.status === 401) {
        handleLogout()
        return
      }

      // Handle unsuccessful API response
      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Document search failed.'
        )
      }

      // Store actual search results
      setSearchResults(data.results || [])
      setHasSearched(true)

    } catch (err) {
      setSearchError(
        err.message || 'Unable to search documents.'
      )
    } finally {
      setSearchLoading(false)
    }
  }

  // ==========================================
  // 9. LOGOUT FUNCTION
  // ==========================================

  function handleLogout() {
    sessionStorage.removeItem('access_token')
    sessionStorage.removeItem('tenant_id')
    sessionStorage.removeItem('user_role')

    setUser(null)
    setDocuments([])

    // Clear search state
    setSearchQuery('')
    setSearchResults([])
    setSearchError('')
    setHasSearched(false)
  }

  // ==========================================
  // 10. LOGIN SCREEN
  // ==========================================

  if (!user) {
    return (
      <div className="login-page">
        <div className="login-card">

          <div className="brand">
            <div className="brand-icon">R</div>
            <h1>Production-RAG</h1>
            <p>Intelligent Document Search & AI Assistant</p>
          </div>

          <div className="login-heading">
            <h2>Welcome back</h2>
            <p>Sign in to access your workspace.</p>
          </div>

          <form onSubmit={handleLogin}>

            <div className="form-group">
              <label htmlFor="email">Email address</label>

              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>

              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={12}
                maxLength={128}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="tenantId">
                Workspace / Tenant ID
              </label>

              <input
                id="tenantId"
                type="text"
                placeholder="Enter your tenant UUID"
                value={tenantId}
                onChange={(event) => setTenantId(event.target.value)}
                required
              />
            </div>

            {error && (
              <div className="message error">
                {error}
              </div>
            )}

            <button type="submit" disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in'}
            </button>

          </form>

          <div className="login-footer">
            <p>Secure multi-tenant document intelligence</p>
          </div>

        </div>
      </div>
    )
  }

  // ==========================================
  // 11. AUTHENTICATED DASHBOARD
  // ==========================================

  return (
    <div className="dashboard">

      {/* Dashboard Header */}

      <header className="dashboard-header">

        <div>
          <h1>Production-RAG</h1>
          <p>Document Intelligence Dashboard</p>
        </div>

        <div className="user-info">
          <span>{user.role}</span>

          <button onClick={handleLogout}>
            Logout
          </button>
        </div>

      </header>

      {/* Dashboard Main Content */}

      <main className="dashboard-content">

        <h2>Welcome to your workspace</h2>

        <p>Monitor your document intelligence system.</p>

        {/* System Statistics */}

        <section className="stats-grid">

          <div className="stat-card">
            <h3>Total Documents</h3>
            <strong>{documents.length}</strong>
          </div>

          <div className="stat-card">
            <h3>API Health</h3>
            <strong>{health}</strong>
          </div>

          <div className="stat-card">
            <h3>System Readiness</h3>
            <strong>{readiness}</strong>
          </div>

        </section>


        {/* Document Search */}

        <section className="search-section">

          <h2>Search Your Documents</h2>

          <p>
            Ask a question and find relevant information
            from your workspace documents.
          </p>

          <form onSubmit={handleSearch} className="search-form">

            <input
              type="text"
              placeholder="Enter your question..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
            />

            <button type="submit" disabled={searchLoading}>
              {searchLoading ? 'Searching...' : 'Search'}
            </button>

          </form>

          {searchError && (
            <div className="message error">
              {searchError}
            </div>
          )}

          {hasSearched && searchResults.length === 0 && (
            <p className="no-results">
              No relevant documents found.
            </p>
          )}

          {searchResults.length > 0 && (
            <div className="search-results">

              <h3>Search Results ({searchResults.length})</h3>

              {searchResults.map((result, index) => (

                <article
                  className="search-result-card"
                  key={result.chunk_id || index}
                >

                  <h4>{result.filename}</h4>

                  <p className="result-text">
                    {result.text}
                  </p>

                  <div className="search-result-meta">

                    <span>
                      Pages: {result.start_page} - {result.end_page}
                    </span>

                    <span>
                      Section: {result.section_type || 'General'}
                    </span>

                    {typeof result.reranker_score === 'number' && (
                      <span>
                        Relevance: {result.reranker_score.toFixed(3)}
                      </span>
                    )}

                  </div>

                </article>

              ))}

            </div>
          )}

        </section>


        {/* Dashboard Error */}

        {dashboardError && (
          <div className="message error">
            {dashboardError}
          </div>
        )}

        {/* Document List */}

        <section className="documents-section">

          <h2>Your Documents</h2>

          {documents.length === 0 ? (
            <p>No documents found in this workspace.</p>
          ) : (
            <div className="document-list">

              {documents.map((document) => (
                <div
                  className="document-item"
                  key={document.document_id}
                >

                  <div>
                    <strong>{document.filename}</strong>
                    <p>{document.document_id}</p>
                  </div>

                  <span className={`status ${document.status}`}>
                    {document.status}
                  </span>

                </div>
              ))}

            </div>
          )}

        </section>

      </main>

    </div>
  )
}

export default App
