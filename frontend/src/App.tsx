import './App.css'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { RouterProvider } from 'react-router-dom'
import { queryClient } from './api/queryClient'
import { routes } from './routes/routes'

function App() {

  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={routes}/>
      <ReactQueryDevtools/>
    </QueryClientProvider>
  )
}

export default App
