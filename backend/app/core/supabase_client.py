# Placeholder file to avoid import errors
# Supabase functionality has been removed

# Mock object that does nothing
class DummySupabase:
    def table(self, _):
        return self
    
    def insert(self, _):
        return self
    
    def execute(self):
        return {"data": [], "status": "not_implemented"}

# Provide a dummy instance for compatibility
supabase = DummySupabase()
