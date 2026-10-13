"""
Real-time Collaboration Module for MedScan AI
Handles multi-doctor collaboration sessions via WebSocket
"""

from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import session, request
import uuid
from datetime import datetime

# Collaboration sessions storage (in-memory for demo)
sessions = {}
session_users = {}

def init_collab(socketio: SocketIO):
    """Initialize collaboration endpoints"""
    
    @socketio.on('join_collab')
    def on_join_collab(data):
        """User joins a collaboration session"""
        session_id = data.get('session_id', str(uuid.uuid4())[:8])
        username = data.get('username', 'Anonymous')
        modality = data.get('modality', 'xray')
        
        # Store session info
        if session_id not in sessions:
            sessions[session_id] = {
                'modality': modality,
                'prediction': None,
                'annotations': {},
                'chat': [],
                'created_at': datetime.now().isoformat(),
                'users': []
            }
        
        join_room(session_id)
        session_users[request.sid] = {
            'username': username,
            'session_id': session_id,
            'joined_at': datetime.now().isoformat()
        }
        
        # Add user to session
        sessions[session_id]['users'].append({
            'sid': request.sid,
            'username': username,
            'role': 'doctor'  # Could be patient/doctor/admin
        })
        
        emit('user_joined', {'username': username}, room=session_id)
        emit('session_data', sessions[session_id], room=session_id)
        
        emit('status', {
            'message': f'Joined session {session_id}',
            'users_online': len(sessions[session_id]['users'])
        })
    
    @socketio.on('leave_collab')
    def on_leave_collab():
        """User leaves collaboration session"""
        if request.sid in session_users:
            session_id = session_users[request.sid]['session_id']
            username = session_users[request.sid]['username']
            
            leave_room(session_id)
            
            # Remove from session users
            if session_id in sessions:
                sessions[session_id]['users'] = [
                    u for u in sessions[session_id]['users'] 
                    if u['sid'] != request.sid
                ]
                emit('user_left', {'username': username}, room=session_id)
            
            del session_users[request.sid]
    
    @socketio.on('chat_message')
    def on_chat_message(data):
        """Send chat message to session"""
        if request.sid in session_users:
            session_id = session_users[request.sid]['session_id']
            username = session_users[request.sid]['username']
            
            message_data = {
                'username': username,
                'message': data.get('message', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            if session_id in sessions:
                sessions[session_id]['chat'].append(message_data)
                emit('new_message', message_data, room=session_id)
    
    @socketio.on('annotation')
    def on_annotation(data):
        """Share annotation on medical image"""
        if request.sid in session_users:
            session_id = session_users[request.sid]['session_id']
            username = session_users[request.sid]['username']
            
            annotation_data = {
                'id': str(uuid.uuid4()),
                'username': username,
                'type': data.get('type'),  # 'highlight', 'draw', 'text'
                'position': data.get('position'),
                'color': data.get('color', '#ff0000'),
                'text': data.get('text'),
                'timestamp': datetime.now().isoformat()
            }
            
            if session_id in sessions:
                sessions[session_id]['annotations'][annotation_data['id']] = annotation_data
                emit('new_annotation', annotation_data, room=session_id)
    
    @socketio.on('update_prediction')
    def on_update_prediction(data):
        """Update shared prediction data"""
        if request.sid in session_users:
            session_id = session_users[request.sid]['session_id']
            
            if session_id in sessions:
                sessions[session_id]['prediction'] = data.get('prediction')
                emit('prediction_updated', data.get('prediction'), room=session_id)
    
    @socketio.on('disconnect')
    def on_disconnect():
        """Handle disconnect"""
        on_leave_collab()
    
    return socketio

