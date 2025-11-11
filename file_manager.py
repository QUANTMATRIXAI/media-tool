"""
File Manager - Helper functions for managing user files across modules with project support
"""

import streamlit as st
import io
from database import db


def get_current_project(module):
    """Get the current project for a module"""
    project_key = f'{module}_current_project'
    if project_key not in st.session_state:
        # Try to get the last accessed project
        if 'username' in st.session_state:
            projects = db.get_projects(st.session_state['username'], module)
            if projects:
                st.session_state[project_key] = projects[0]['name']  # Most recently accessed
            else:
                st.session_state[project_key] = None
    return st.session_state.get(project_key)


def set_current_project(module, project_name):
    """Set the current project for a module"""
    project_key = f'{module}_current_project'
    st.session_state[project_key] = project_name
    if 'username' in st.session_state and project_name:
        db.update_project_access(st.session_state['username'], module, project_name)


def save_uploaded_file(module, file_key, uploaded_file):
    """Save an uploaded file to the database"""
    if uploaded_file is not None and 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        
        # Debug info
        if not project_name:
            st.error(f"❌ No project selected! Cannot save file.")
            return False
        
        try:
            success = db.save_file(username, module, project_name, file_key, uploaded_file)
            if not success:
                st.error(f"❌ Failed to save file to database")
            return success
        except Exception as e:
            st.error(f"❌ Error saving file: {str(e)}")
            return False
    else:
        if 'username' not in st.session_state:
            st.error("❌ Not logged in!")
        return False


def load_saved_file(module, file_key):
    """Load a previously saved file from the database"""
    if 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        if project_name:
            file_data = db.get_file(username, module, project_name, file_key)
            
            if file_data:
                # Create a file-like object from the saved data
                file_obj = io.BytesIO(file_data['file_data'])
                file_obj.name = file_data['file_name']
                return file_obj
    return None


def get_user_files_summary(module):
    """Get summary of all saved files for current user in a module and project"""
    if 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        if project_name:
            return db.get_all_files(username, module, project_name)
    return {}


def delete_saved_file(module, file_key):
    """Delete a saved file"""
    if 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        if project_name:
            return db.delete_file(username, module, project_name, file_key)
    return False


def save_module_progress(module, progress_dict):
    """Save progress/settings for a module and project"""
    if 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        if project_name:
            return db.save_progress(username, module, project_name, progress_dict)
    return False


def load_module_progress(module):
    """Load progress/settings for a module and project"""
    if 'username' in st.session_state:
        username = st.session_state['username']
        project_name = get_current_project(module)
        if project_name:
            progress = db.get_progress(username, module, project_name)
            if progress:
                return progress['data']
    return None


def smart_file_uploader(label, file_types, module, file_key, **kwargs):
    """
    Enhanced file uploader that automatically saves/loads files per user
    
    Args:
        label: Label for the file uploader
        file_types: List of accepted file types
        module: Module name ('eda', 'modeling', 'optimizer')
        file_key: Unique key for this file
        **kwargs: Additional arguments for st.file_uploader
    
    Returns:
        Uploaded file object (either new upload or loaded from database)
    """
    # Check if user has a saved file
    saved_file = load_saved_file(module, file_key)
    
    # Show saved file info if exists
    if saved_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📁 Using saved file: **{saved_file.name}**")
        with col2:
            if st.button("🗑️", key=f"delete_{module}_{file_key}", help="Delete saved file"):
                if delete_saved_file(module, file_key):
                    st.success("File deleted!")
                    st.rerun()
    
    # File uploader
    uploaded_file = st.file_uploader(label, type=file_types, key=f"upload_{module}_{file_key}", **kwargs)
    
    # If new file uploaded, save it
    if uploaded_file is not None:
        # Check if this is a new file (different from saved file)
        is_new_file = True
        if saved_file and hasattr(saved_file, 'name') and hasattr(uploaded_file, 'name'):
            is_new_file = saved_file.name != uploaded_file.name
        
        if is_new_file:
            if save_uploaded_file(module, file_key, uploaded_file):
                st.success(f"✅ File '{uploaded_file.name}' saved!")
        
        return uploaded_file
    
    # Return saved file if no new upload
    return saved_file


def show_project_selector(module):
    """Show project selector and management UI"""
    if 'username' not in st.session_state:
        return None
    
    username = st.session_state['username']
    projects = db.get_projects(username, module)
    
    st.markdown("### 📁 Project Management")
    
    # Create new project section
    with st.expander("➕ Create New Project", expanded=len(projects) == 0):
        new_project_name = st.text_input("Project Name", key=f"{module}_new_project_name")
        new_project_desc = st.text_area("Description (optional)", key=f"{module}_new_project_desc")
        
        if st.button("Create Project", key=f"{module}_create_project", type="primary"):
            if new_project_name:
                if db.create_project(username, module, new_project_name, new_project_desc):
                    set_current_project(module, new_project_name)
                    st.success(f"✅ Project '{new_project_name}' created!")
                    st.rerun()
                else:
                    st.error("❌ Project already exists or error occurred")
            else:
                st.warning("Please enter a project name")
    
    # Select existing project
    if projects:
        st.markdown("**Select Project:**")
        current_project = get_current_project(module)
        
        project_names = [p['name'] for p in projects]
        current_index = project_names.index(current_project) if current_project in project_names else 0
        
        selected_project = st.selectbox(
            "Choose a project to work on:",
            project_names,
            index=current_index,
            key=f"{module}_project_selector",
            label_visibility="collapsed"
        )
        
        if selected_project != current_project:
            set_current_project(module, selected_project)
            st.rerun()
        
        # Show project info
        project_info = next((p for p in projects if p['name'] == selected_project), None)
        if project_info:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📅 Created: {project_info['created_at'][:10]}")
            with col2:
                st.caption(f"🕒 Last accessed: {project_info['last_accessed'][:10]}")
            
            if project_info['description']:
                st.info(f"📝 {project_info['description']}")
        
        # Delete project button
        if st.button("🗑️ Delete Current Project", key=f"{module}_delete_project"):
            if db.delete_project(username, module, selected_project):
                st.success(f"✅ Project '{selected_project}' deleted!")
                set_current_project(module, None)
                st.rerun()
    else:
        st.info("👆 Create your first project to get started!")
    
    st.divider()
    return get_current_project(module)


def show_file_management_panel(module):
    """Show a panel with all saved files for the current module and project"""
    files = get_user_files_summary(module)
    
    if files:
        with st.expander(f"📂 Saved Files ({len(files)})", expanded=False):
            st.caption("💾 All files are automatically saved to your project")
            for file_key, file_info in files.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(f"🔑 {file_key}")
                with col2:
                    st.text(f"📄 {file_info['file_name']}")
                with col3:
                    if st.button("🗑️", key=f"del_panel_{module}_{file_key}"):
                        if delete_saved_file(module, file_key):
                            st.success("Deleted!")
                            st.rerun()
    else:
        st.caption("💡 Upload files above - they'll be saved automatically!")
